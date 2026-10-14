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

import math
from collections import Counter
from dataclasses import replace

import pytest
import torch

from flashinfer.experimental.nvfp4_mla_decode import cake_backend
from flashinfer.experimental.nvfp4_mla_decode.cake_backend import (
    BALANCED_MIN_KV,
    BOUNDARY_COST_TILES,
    CLUSTER_PAIR,
    DSV4_Q_LEN,
    FLAG_DIRECT_OUT,
    FLAG_SEED_SINK,
    HEAD_DIM,
    ITEM_FIELDS,
    MAX_SPLITS,
    PAGE_SIZE,
    ROW_BYTES,
    ROWS_PER_TILE,
    SF_ROW_BYTES,
    SF_VEC,
    SUPPORTED_COMPUTE_CAPABILITIES,
    V_HALVES,
    build_work_plan,
    check_pairs,
    host_tables,
    kv_pages,
    kv_tiles,
    max_nvfp4_mla_decode_workspace_size,
    nvfp4_mla_decode_workspace_size,
    quantize_nvfp4,
    token_splits,
    work_table_rows,
    workspace_layout,
)
from flashinfer.mla import prepare_nvfp4_batch_decode_with_kv_cache_mla

ATOL = RTOL = 0.1
REL_L2_MAX = 0.05
LSE_ATOL = LSE_RTOL = 0.05
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


# ---------------------------------------------------------------------------
# Torch reference: FP32 attention over exactly dequantized NVFP4 operands
# ---------------------------------------------------------------------------


def dequantize_nvfp4(packed, scale):
    """Exact FP32 decode of packed E2M1 codes with UE4M3 block-16 scales."""
    lo = packed & 0x0F
    hi = packed >> 4
    codes = torch.stack((lo, hi), dim=-1).reshape(
        *packed.shape[:-1], packed.shape[-1] * 2
    )
    table = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=packed.device)
    magnitude = table[(codes & 0x7).long()]
    sign = torch.where((codes & 0x8) != 0, -1.0, 1.0)
    values = magnitude * sign
    blocks = values.reshape(*values.shape[:-1], values.shape[-1] // SF_VEC, SF_VEC)
    scale = scale.view(torch.float8_e4m3fn).float()
    return (blocks * scale.unsqueeze(-1)).reshape(*values.shape)


def _gather_dequant(cache, scale, block_table_row, kv_len):
    """Dequantize the first ``kv_len`` tokens of one request: [kv_len, 512]."""
    pages = block_table_row[: kv_pages(kv_len)].long()
    rows = dequantize_nvfp4(cache[pages], scale[pages])  # [pages, page, D]
    return rows.reshape(-1, rows.shape[-1])[:kv_len]


def reference(
    query,
    query_scale,
    kv_cache,
    kv_scale,
    block_tables,
    kv_lens,
    q_len,
    sm_scale,
    sinks,
):
    """Returns ``(O bf16 [total_q, H, 512], LSE f32 [total_q, H])``."""
    batch = len(kv_lens)
    q_all = dequantize_nvfp4(query, query_scale)  # [total_q, H, D]
    num_heads = q_all.shape[1]
    O = torch.empty(
        (batch * q_len, num_heads, HEAD_DIM), dtype=torch.float32, device=q_all.device
    )
    LSE = torch.empty(
        (batch * q_len, num_heads), dtype=torch.float32, device=q_all.device
    )
    for b in range(batch):
        kv_len = kv_lens[b]
        k = _gather_dequant(kv_cache, kv_scale, block_tables[b], kv_len)  # [kv_len, D]
        v = k
        q = q_all[b * q_len : (b + 1) * q_len]  # [q_len, H, D]
        logits = torch.einsum("rhd,nd->hrn", q, k) * sm_scale  # [H, q_len, kv_len]
        positions = torch.arange(kv_len, device=q.device)
        row_limit = kv_len - q_len + torch.arange(q_len, device=q.device) + 1
        mask = positions[None, :] < row_limit[:, None]  # [q_len, kv_len]
        logits = logits.masked_fill(~mask[None], float("-inf"))
        row_max = logits.amax(dim=-1)  # [H, q_len]
        if sinks is not None:
            sink = sinks[:, None].expand(num_heads, q_len)
            row_max = torch.maximum(row_max, sink)
        probs = torch.exp(logits - row_max[..., None])
        denom = probs.sum(dim=-1)
        if sinks is not None:
            denom = denom + torch.exp(sink - row_max)
        out = torch.einsum("hrn,nd->hrd", probs, v) / denom[..., None]
        lse = row_max + torch.log(denom)
        O[b * q_len : (b + 1) * q_len] = out.permute(1, 0, 2)
        LSE[b * q_len : (b + 1) * q_len] = lse.permute(1, 0)
    return O.to(torch.bfloat16), LSE


def make_inputs(kv_lens, num_heads, *, q_len, enable_sink, device, seed=0):
    """Deterministic NVFP4 paged decode inputs with a peaked softmax."""
    gen = torch.Generator(device=device).manual_seed(seed)
    batch = len(kv_lens)
    pages_per_seq = [kv_pages(kv) for kv in kv_lens]
    total_pages = sum(pages_per_seq)
    max_pages = max(pages_per_seq)
    k_full = torch.randn(
        (total_pages, PAGE_SIZE, HEAD_DIM), generator=gen, device=device
    )
    permutation = torch.randperm(total_pages, generator=gen, device=device).to(
        torch.int32
    )
    block_tables = torch.zeros((batch, max_pages), dtype=torch.int32, device=device)
    offset = 0
    for b, count in enumerate(pages_per_seq):
        block_tables[b, :count] = permutation[offset : offset + count]
        offset += count
    q = torch.randn((batch * q_len, num_heads, HEAD_DIM), generator=gen, device=device)
    for b in range(batch):
        for i in range(q_len):
            visible = kv_lens[b] - q_len + i + 1
            target = int(
                torch.randint(0, visible, (1,), generator=gen, device=device).item()
            )
            page = int(block_tables[b, target // PAGE_SIZE].item())
            q[b * q_len + i] += 0.2 * k_full[page, target % PAGE_SIZE]
    query, query_scale = quantize_nvfp4(q)
    kv_cache, kv_scale = quantize_nvfp4(k_full)
    sinks = None
    if enable_sink:
        sinks = torch.randn((num_heads,), generator=gen, device=device) * 0.5 + 1.0
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    return dict(
        query=query,
        query_scale=query_scale,
        kv_cache=kv_cache,
        kv_scale=kv_scale,
        block_tables=block_tables,
        seq_lens=seq_lens,
        sinks=sinks,
        q_len=q_len,
        sm_scale=HEAD_DIM**-0.5,
    )


def check_outputs(out, lse, ref_out, ref_lse):
    assert torch.isfinite(out.float()).all()
    torch.testing.assert_close(out, ref_out, atol=ATOL, rtol=RTOL)
    rel = (out.float() - ref_out.float()).norm(dim=-1) / ref_out.float().norm(
        dim=-1
    ).clamp_min(1e-6)
    assert float(rel.max()) <= REL_L2_MAX, float(rel.max())
    assert torch.isfinite(lse).all()
    torch.testing.assert_close(lse, ref_lse, atol=LSE_ATOL, rtol=LSE_RTOL)


def assert_plan_well_formed(plan, kv_lens, *, q_len, num_heads, enable_sink):
    """Behavioural invariants every work plan must satisfy.

    Every (request, row tile, pair row) covers its pipeline tiles exactly
    once with contiguous, ordered splits; flags follow the split structure;
    unit ranges partition the table into whole row pairs of one piece; the
    launch grid is one cluster per unit.
    """
    m_tiles = math.ceil(q_len * num_heads / ROWS_PER_TILE)
    covered = Counter()
    pieces = {}
    for it in plan.items:
        b, m_tile, pair_row, start, end, split, num_splits, flags = it
        assert 0 <= b < len(kv_lens) and 0 <= m_tile < m_tiles and pair_row in (0, 1)
        assert 0 <= start < end <= kv_tiles(kv_lens[b]), it
        assert 0 <= split < num_splits <= MAX_SPLITS, it
        assert ((flags & FLAG_DIRECT_OUT) != 0) == (num_splits == 1), it
        assert ((flags & FLAG_SEED_SINK) != 0) == (enable_sink and split == 0), it
        for t in range(start, end):
            covered[(b, m_tile, pair_row, t)] += 1
        pieces.setdefault((b, m_tile, pair_row), {})[split] = (start, end, num_splits)
    want = sum(kv_tiles(kv) for kv in kv_lens) * m_tiles * V_HALVES
    assert len(covered) == want and max(covered.values()) == 1
    for key, splits in pieces.items():
        counts = {s[2] for s in splits.values()}
        assert counts == {len(splits)} and sorted(splits) == list(range(len(splits))), (
            key
        )
        ordered = [splits[s] for s in sorted(splits)]
        assert ordered[0][0] == 0 and ordered[-1][1] == kv_tiles(kv_lens[key[0]])
        for (_, e0, _), (s1, _, _) in zip(ordered, ordered[1:], strict=False):
            assert e0 == s1
    assert plan.max_splits == max(len(s) for s in pieces.values())
    assert plan.unit_first[0] == 0 and plan.unit_first[-1] == plan.num_items
    assert all(
        a < b for a, b in zip(plan.unit_first, plan.unit_first[1:], strict=False)
    )
    check_pairs(plan)
    assert plan.grid == (CLUSTER_PAIR * plan.num_units, 1, 1)


# ---------------------------------------------------------------------------
# Host-only tests
# ---------------------------------------------------------------------------


def test_public_entry_point_is_experimental():
    assert prepare_nvfp4_batch_decode_with_kv_cache_mla.is_experimental is True


def test_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        prepare_nvfp4_batch_decode_with_kv_cache_mla(
            None, None, None, None, None, None, None, sm_scale=1.0, backend="unknown"
        )


def test_rejects_host_tensors_and_bad_shapes():
    inputs = make_inputs([64, 100], 16, q_len=6, enable_sink=False, device="cpu")
    workspace = torch.empty(1 << 20, dtype=torch.uint8)
    with pytest.raises(ValueError, match="CUDA"):
        prepare_nvfp4_batch_decode_with_kv_cache_mla(
            inputs["query"],
            inputs["query_scale"],
            inputs["kv_cache"],
            inputs["kv_scale"],
            inputs["block_tables"],
            inputs["seq_lens"],
            workspace,
            sm_scale=inputs["sm_scale"],
        )
    args = (
        inputs["kv_cache"],
        inputs["kv_scale"],
        inputs["block_tables"],
        inputs["seq_lens"],
    )
    with pytest.raises(ValueError, match="query must be"):
        cake_backend.validate_nvfp4_mla_decode_inputs(
            inputs["query"][..., :128], inputs["query_scale"], *args
        )
    with pytest.raises(ValueError, match="batch \\* q_len"):
        cake_backend.validate_nvfp4_mla_decode_inputs(
            inputs["query"][:-1], inputs["query_scale"][:-1], *args
        )
    with pytest.raises(ValueError, match="int32"):
        cake_backend.validate_nvfp4_mla_decode_inputs(
            inputs["query"],
            inputs["query_scale"],
            inputs["kv_cache"],
            inputs["kv_scale"],
            inputs["block_tables"].to(torch.int64),
            inputs["seq_lens"],
        )


@pytest.mark.parametrize("q_len", [1, 6])
def test_q_len_is_derived_from_query_rows(q_len):
    inputs = make_inputs(
        [64, 100, 130], 64, q_len=q_len, enable_sink=False, device="cpu"
    )
    batch, num_heads, num_pages, got = cake_backend.validate_nvfp4_mla_decode_inputs(
        inputs["query"],
        inputs["query_scale"],
        inputs["kv_cache"],
        inputs["kv_scale"],
        inputs["block_tables"],
        inputs["seq_lens"],
    )
    assert (batch, num_heads, got) == (3, 64, q_len)
    assert num_pages == sum(kv_pages(kv) for kv in [64, 100, 130])


@pytest.mark.parametrize("kv_lens,num_heads,q_len", [([512, 300], 8, 6), ([64], 64, 1)])
def test_rejects_fewer_than_128_query_rows(kv_lens, num_heads, q_len):
    inputs = make_inputs(
        kv_lens, num_heads, q_len=q_len, enable_sink=False, device="cpu"
    )
    with pytest.raises(ValueError, match="must be >= 128"):
        cake_backend.validate_nvfp4_mla_decode_inputs(
            inputs["query"],
            inputs["query_scale"],
            inputs["kv_cache"],
            inputs["kv_scale"],
            inputs["block_tables"],
            inputs["seq_lens"],
        )


def unit_model_costs(plan):
    """Per-unit model cost ``tiles + BOUNDARY_COST_TILES * (pieces - 1)`` of a balanced plan.

    Counted over the ``pair_row == 0`` items of each unit (both rows of a
    piece share the tile range); this is the quantity the host plan equalises.
    """
    costs = []
    for a, b in zip(plan.unit_first, plan.unit_first[1:], strict=False):
        pieces = [it for it in plan.items[a:b] if it[2] == 0]
        tiles = sum(it[4] - it[3] for it in pieces)
        costs.append(tiles + BOUNDARY_COST_TILES * (len(pieces) - 1))
    return costs


def assert_cost_balanced(plan, kv_lens, *, q_len, num_heads, num_sms):
    """The balanced partition is cost-balanced, not tile-balanced.

    The greedy fill behind ``cost_balanced_unit_bounds`` closes a unit once the
    next tile (plus its boundary cost when it opens a new piece) would exceed
    the per-unit bound, so every unit but the last is within one tile plus one
    boundary cost of the most expensive unit, and the most expensive unit
    costs no more than the equal-tiles partition's worst unit.
    """
    costs = unit_model_costs(plan)
    worst = max(costs)
    assert all(c > worst - (BOUNDARY_COST_TILES + 1) for c in costs[:-1]), costs
    m_tiles = math.ceil(q_len * num_heads / ROWS_PER_TILE)
    seq_lens = [kv_tiles(kv) for kv in kv_lens for _ in range(m_tiles)]
    total = sum(seq_lens)
    units = plan.num_units
    equal_worst = 0.0
    for u in range(units):
        lo, hi = (u * total) // units, ((u + 1) * total) // units
        pos, pieces = 0, 0
        for tiles in seq_lens:
            if pos < hi and pos + tiles > lo:
                pieces += 1
            pos += tiles
        equal_worst = max(
            equal_worst, (hi - lo) + BOUNDARY_COST_TILES * max(pieces - 1, 0)
        )
    assert worst <= equal_worst + 1e-9, (worst, equal_worst)
    assert plan.num_units <= num_sms // CLUSTER_PAIR


# Plan geometry of the shapes the GPU tests run (128-token pipeline tiles;
# cluster of two CTAs per unit; balanced schedule over num_sms // 2 units;
# uniform schedule pairs consecutive rows). Columns: schedule, kv_lens, q_len,
# num_heads, num_sms, tiles_per_split, enable_sink -> (num_items, num_units,
# grid_x, max_splits, tiles_per_split). The item list itself is checked by
# ``assert_plan_well_formed``.
PLAN_EXPECTATIONS = [
    ("balanced", [256, 300], 6, 64, 148, None, True, (12, 1, 2, 1, None)),
    # cost-balanced partition: the optimum needs 72 of the 74 units
    ("balanced", [8192] * 4, 6, 64, 148, None, False, (144, 72, 144, 6, None)),
    # contract row partial_pages_h8
    ("balanced", [1000, 8192, 5000], 6, 8, 148, None, True, (28, 14, 28, 8, None)),
    ("balanced", [131072, 64], 1, 64, 148, None, False, (76, 37, 74, 37, None)),
    # contract row bs32_q6_kv8k (auto -> balanced at the 8K threshold)
    ("auto", [8192] * 32, 6, 64, 148, None, True, (324, 74, 148, 2, None)),
    ("uniform", [64, 4096], 6, 64, 148, 16, False, (18, 9, 18, 2, 16)),
    # contract row smoke, forced to two splits (tiles_per_split = ceil(3 / 2))
    ("auto", [256, 300], 6, 64, 148, 2, True, (18, 9, 18, 2, 2)),
    ("uniform", [8192] * 32, 6, 64, 148, None, False, (192, 96, 192, 1, 64)),
    ("uniform", [777, 65], 1, 64, 148, None, True, (4, 2, 4, 1, 7)),
    # contract row no_sink_q1 (auto -> uniform)
    ("auto", [64, 65, 4096, 777], 1, 64, 148, None, False, (8, 4, 8, 1, 32)),
    ("uniform", [16384, 12000], 6, 64, 148, None, False, (12, 6, 12, 1, 128)),
]


@pytest.mark.parametrize(
    "schedule,kv_lens,q_len,num_heads,num_sms,tiles_per_split,enable_sink,expected",
    PLAN_EXPECTATIONS,
)
def test_plan_geometry_and_invariants(
    schedule, kv_lens, q_len, num_heads, num_sms, tiles_per_split, enable_sink, expected
):
    plan = build_work_plan(
        kv_lens,
        num_heads=num_heads,
        num_sms=num_sms,
        q_len=q_len,
        enable_sink=enable_sink,
        schedule=schedule,
        tiles_per_split=tiles_per_split,
    )
    got = (
        plan.num_items,
        plan.num_units,
        plan.grid[0],
        plan.max_splits,
        plan.tiles_per_split,
    )
    assert got == expected
    assert_plan_well_formed(
        plan, kv_lens, q_len=q_len, num_heads=num_heads, enable_sink=enable_sink
    )
    if schedule == "auto":
        assert plan.schedule == (
            "balanced" if max(kv_lens) >= BALANCED_MIN_KV else "uniform"
        )
    if plan.schedule == "uniform":
        assert plan.unit_first == tuple(range(0, plan.num_items + 1, CLUSTER_PAIR))
        assert all(it[4] - it[3] <= plan.tiles_per_split for it in plan.items)
    else:
        assert_cost_balanced(
            plan, kv_lens, q_len=q_len, num_heads=num_heads, num_sms=num_sms
        )


def test_balanced_units_are_contiguous_in_request_major_order():
    plan = build_work_plan(
        [256, 300],
        num_heads=64,
        num_sms=148,
        q_len=6,
        enable_sink=True,
        schedule="balanced",
    )
    assert plan.unit_first == (0, 12)
    plan = build_work_plan(
        [8192] * 32,
        num_heads=64,
        num_sms=148,
        q_len=6,
        enable_sink=True,
        schedule="balanced",
    )
    # Every unit receives whole pairs of one piece; the request-major order of
    # the pieces is preserved across unit boundaries.
    keys = [(it[0], it[1], it[3]) for it in plan.items[::V_HALVES]]
    assert keys == sorted(keys)
    assert all(
        (b - a) % V_HALVES == 0
        for a, b in zip(plan.unit_first, plan.unit_first[1:], strict=False)
    )


def test_pair_invariant_holds_and_is_enforced():
    plan = build_work_plan(
        [1000, 8192, 5000], num_heads=8, num_sms=148, q_len=6, enable_sink=True
    )
    check_pairs(plan)
    items = list(plan.items)
    items[0], items[1] = items[1], items[0]
    with pytest.raises(ValueError, match="v_half pair"):
        check_pairs(replace(plan, items=tuple(items)))
    with pytest.raises(ValueError, match="even count"):
        check_pairs(replace(plan, unit_first=(0, 1, plan.num_items)))


def test_uniform_plan_partitions_every_request():
    plan = build_work_plan(
        [256, 300],
        num_heads=64,
        num_sms=148,
        q_len=6,
        enable_sink=True,
        schedule="uniform",
    )
    m_tiles = math.ceil(6 * 64 / ROWS_PER_TILE)
    for b, kv in enumerate([256, 300]):
        ranges = sorted({(it[3], it[4]) for it in plan.items if it[0] == b})
        assert ranges[0][0] == 0 and ranges[-1][1] == kv_tiles(kv)
        for (_, e0), (s1, _) in zip(ranges, ranges[1:], strict=False):
            assert e0 == s1
        assert (
            sum(1 for it in plan.items if it[0] == b)
            == len(ranges) * m_tiles * V_HALVES
        )
    assert all(
        (it[7] & FLAG_SEED_SINK) == (FLAG_SEED_SINK if it[5] == 0 else 0)
        for it in plan.items
    )
    assert all(((it[7] & FLAG_DIRECT_OUT) != 0) == (it[6] == 1) for it in plan.items)


@pytest.mark.parametrize(
    "kv_len,want_splits", [(8192, 1), (16384, 2), (32768, 2), (65536, 2), (131072, 2)]
)
def test_uniform_split_policy_matches_measured_winners(kv_len, want_splits):
    plan = build_work_plan(
        [kv_len] * 32, num_heads=64, num_sms=148, q_len=6, schedule="uniform"
    )
    assert plan.max_splits == want_splits
    assert all(it[4] - it[3] >= 8 for it in plan.items)


@pytest.mark.parametrize(
    "kv_lens,num_heads,num_sms",
    [
        ([131072] * 32, 64, 148),
        ([8192] * 32, 64, 148),
        ([256, 300], 64, 148),
        ([1000, 8192, 5000], 8, 148),
        ([131072], 64, 148),
    ],
)
def test_balanced_plan_covers_every_page_once(kv_lens, num_heads, num_sms):
    plan = build_work_plan(
        kv_lens,
        num_heads=num_heads,
        num_sms=num_sms,
        q_len=6,
        enable_sink=True,
        schedule="balanced",
    )
    assert plan.max_splits <= MAX_SPLITS and plan.num_units <= num_sms // CLUSTER_PAIR
    assert_plan_well_formed(
        plan, kv_lens, q_len=6, num_heads=num_heads, enable_sink=True
    )
    assert_cost_balanced(plan, kv_lens, q_len=6, num_heads=num_heads, num_sms=num_sms)


def test_auto_schedule_threshold():
    assert (
        build_work_plan(
            [BALANCED_MIN_KV - 1] * 32, num_heads=64, num_sms=148, q_len=6
        ).schedule
        == "uniform"
    )
    assert (
        build_work_plan(
            [BALANCED_MIN_KV] * 32, num_heads=64, num_sms=148, q_len=6
        ).schedule
        == "balanced"
    )
    with pytest.raises(ValueError, match="kv_len >= q_len"):
        build_work_plan([5], num_heads=64, num_sms=148, q_len=6)


def test_work_table_keeps_direct_out_on_single_split_items():
    single = build_work_plan(
        [64, 65, 4096, 777],
        num_heads=64,
        num_sms=148,
        q_len=1,
        schedule="uniform",
        tiles_per_split=4096,
    )
    assert single.max_splits == 1
    assert bool((work_table_rows(single)[:, 7] & FLAG_DIRECT_OUT).all())
    mixed = build_work_plan(
        [64, 4096],
        num_heads=64,
        num_sms=148,
        q_len=6,
        schedule="uniform",
        tiles_per_split=16,
    )
    table = work_table_rows(mixed)
    assert mixed.max_splits == 2 and table.shape == (mixed.num_items, ITEM_FIELDS)
    # The single-split request keeps the flag, the split request does not.
    direct = (table[:, 7] & FLAG_DIRECT_OUT) != 0
    assert torch.equal(direct, table[:, 6] == 1)
    assert bool(direct[table[:, 0] == 0].all()) and not bool(
        direct[table[:, 0] == 1].any()
    )
    assert torch.equal(table[:, 0].unique(), torch.tensor([0, 1], dtype=torch.int32))


def _q_indptr(batch, q_len):
    return [b * q_len for b in range(batch + 1)]


@pytest.mark.parametrize("num_sms", [148, 152, 160])
def test_token_splits_follow_the_items_of_a_mixed_balanced_plan(num_sms):
    # bs32 / q_len 6 / 64 heads at 8K: the balanced partition splits per
    # (request, row tile), so some row tiles run as a single item while the
    # rest use two splits -- one request's tokens carry different counts.
    batch, q_len, heads = 32, 6, 64
    plan = build_work_plan(
        [8192] * batch,
        num_heads=heads,
        num_sms=num_sms,
        q_len=q_len,
        schedule="balanced",
    )
    ts = token_splits(
        plan,
        _q_indptr(batch, q_len),
        q_len=q_len,
        num_heads=heads,
        total_q=batch * q_len,
    )
    tokens_per_tile = ROWS_PER_TILE // heads
    for it in plan.items:
        assert bool(it[7] & FLAG_DIRECT_OUT) == (it[6] == 1), it
        for t in range(
            it[1] * tokens_per_tile, min((it[1] + 1) * tokens_per_tile, q_len)
        ):
            assert ts[it[0] * q_len + t] == it[6], (it, t)
    assert ts.count(1) > 0 and max(ts) == plan.max_splits == 2
    assert any(
        len({ts[b * q_len + t] for t in range(q_len)}) > 1 for b in range(batch)
    ), "expected a request whose tokens carry different split counts"
    table = work_table_rows(plan)
    assert torch.equal((table[:, 7] & FLAG_DIRECT_OUT) != 0, table[:, 6] == 1)


def test_token_splits_match_uniform_plans():
    batch, q_len, heads = 32, 6, 64
    for kv in (4096, 8192, 32768, 131072):
        plan = build_work_plan(
            [kv] * batch, num_heads=heads, num_sms=148, q_len=q_len, schedule="uniform"
        )
        ts = token_splits(
            plan,
            _q_indptr(batch, q_len),
            q_len=q_len,
            num_heads=heads,
            total_q=batch * q_len,
        )
        assert ts == [plan.max_splits] * (batch * q_len)


def test_token_splits_reject_heads_that_straddle_row_tiles():
    heads = 48  # 128 % 48 != 0: a token would span two row tiles with possibly different counts
    plan = build_work_plan(
        [8192] * 4, num_heads=heads, num_sms=16, q_len=6, schedule="balanced"
    )
    with pytest.raises(ValueError, match="divide"):
        token_splits(plan, _q_indptr(4, 6), q_len=6, num_heads=heads, total_q=24)


def test_token_splits_reject_uncovered_tokens_and_inconsistent_flags():
    batch, q_len, heads = 4, 6, 64
    plan = build_work_plan(
        [8192] * batch, num_heads=heads, num_sms=16, q_len=q_len, schedule="balanced"
    )
    kw = dict(q_len=q_len, num_heads=heads, total_q=batch * q_len)
    token_splits(plan, _q_indptr(batch, q_len), **kw)
    missing = replace(plan, items=tuple(it for it in plan.items if it[0] != 2))
    with pytest.raises(ValueError, match="no work item"):
        token_splits(missing, _q_indptr(batch, q_len), **kw)
    first, *rest = plan.items
    flipped = replace(plan, items=(first[:7] + (first[7] ^ FLAG_DIRECT_OUT,), *rest))
    with pytest.raises(ValueError, match="FLAG_DIRECT_OUT"):
        token_splits(flipped, _q_indptr(batch, q_len), **kw)
    # Two items of one (request, row tile) disagreeing on the split count.
    conflicting = replace(
        plan, items=(first[:6] + (first[6] + 1, first[7] & ~FLAG_DIRECT_OUT), *rest)
    )
    with pytest.raises(ValueError, match="split counts"):
        token_splits(conflicting, _q_indptr(batch, q_len), **kw)


def test_workspace_sizing():
    kv_lens = [8192] * 32
    plan = build_work_plan(
        kv_lens, num_heads=64, num_sms=148, q_len=6, enable_sink=True
    )
    layout = workspace_layout(plan, batch=32, num_heads=64, q_len=6)
    total_q = 32 * 6
    assert plan.max_splits == 2
    assert layout["partial_o"][1] == total_q * 64 * plan.max_splits * HEAD_DIM * 2
    assert layout["partial_lse"][1] == total_q * 64 * plan.max_splits * 4
    assert layout["work_table"][1] == plan.num_items * ITEM_FIELDS * 4
    assert layout["unit_first"][1] == (plan.num_units + 1) * 4
    offsets = [
        layout[k][0]
        for k in (
            "partial_o",
            "partial_lse",
            "work_table",
            "unit_first",
            "row_splits",
            "q_indptr",
            "sinks",
        )
    ]
    assert offsets == sorted(offsets) and all(o % 256 == 0 for o in offsets)
    assert (
        nvfp4_mla_decode_workspace_size(
            kv_lens, 64, num_sms=148, q_len=6, enable_sink=True
        )
        == layout["total"]
    )
    assert layout["total"] <= max_nvfp4_mla_decode_workspace_size(32, 64, q_len=6)
    single = build_work_plan([64, 65, 4096, 777], num_heads=64, num_sms=148, q_len=1)
    single_layout = workspace_layout(single, batch=4, num_heads=64, q_len=1)
    assert (
        single.max_splits == 1
        and single_layout["partial_o"][1] == 64 * HEAD_DIM * 2
        and single_layout["partial_lse"][1] == 4
    )
    assert max_nvfp4_mla_decode_workspace_size(
        32, 64, max_splits=1
    ) < max_nvfp4_mla_decode_workspace_size(32, 64, max_splits=2)


def test_quantize_nvfp4_roundtrip_and_saturation():
    x = torch.randn((4, 3, HEAD_DIM))
    packed, scale = quantize_nvfp4(x)
    assert packed.shape == (4, 3, ROW_BYTES) and packed.dtype == torch.uint8
    assert scale.shape == (4, 3, SF_ROW_BYTES) and scale.dtype == torch.uint8
    assert packed.is_contiguous() and scale.is_contiguous()
    decoded = dequantize_nvfp4(packed, scale)
    # The coarsest E2M1 bin (4 -> 6) has a half-step of one normalized unit,
    # so every element is within one decoded block scale of its source.
    block_scale = (
        scale.view(torch.float8_e4m3fn)
        .float()
        .unsqueeze(-1)
        .expand(-1, -1, -1, SF_VEC)
        .reshape_as(x)
    )
    assert ((decoded - x).abs() <= block_scale + 1e-6).all()
    values = torch.tensor([0.0, 1.0, 2688.0, 4096.0, -4096.0], dtype=torch.float32)
    packed, scale = quantize_nvfp4(values[:, None].expand(-1, 32).contiguous())
    scales = scale.view(torch.float8_e4m3fn).float()
    torch.testing.assert_close(
        scales,
        torch.tensor([2.0**-9, 0.171875, 448.0, 448.0, 448.0])[:, None].expand(-1, 2),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        packed[:, 0],
        torch.tensor([0x00, 0x77, 0x77, 0x77, 0xFF], dtype=torch.uint8),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("q_len,enable_sink", [(6, False), (6, True), (1, True)])
def test_reference_matches_masked_softmax(q_len, enable_sink):
    """The ported reference equals a plain softmax with the sink as an extra logit."""
    kv_lens = [70, 130]
    inputs = make_inputs(
        kv_lens, 4, q_len=q_len, enable_sink=enable_sink, device="cpu", seed=3
    )
    out, lse = reference(
        inputs["query"],
        inputs["query_scale"],
        inputs["kv_cache"],
        inputs["kv_scale"],
        inputs["block_tables"],
        kv_lens,
        q_len,
        inputs["sm_scale"],
        inputs["sinks"],
    )
    q_all = dequantize_nvfp4(inputs["query"], inputs["query_scale"])
    for b, kv_len in enumerate(kv_lens):
        k = _gather_dequant(
            inputs["kv_cache"], inputs["kv_scale"], inputs["block_tables"][b], kv_len
        )
        for i in range(q_len):
            visible = kv_len - q_len + i + 1
            logits = (
                q_all[b * q_len + i] @ k[:visible].T * inputs["sm_scale"]
            )  # [H, visible]
            if enable_sink:
                logits = torch.cat([logits, inputs["sinks"][:, None]], dim=-1)
            probs = torch.softmax(logits, dim=-1)
            expected = probs[:, :visible] @ k[:visible]
            torch.testing.assert_close(
                out[b * q_len + i].float(), expected, atol=2e-2, rtol=2e-2
            )
            torch.testing.assert_close(
                lse[b * q_len + i],
                torch.logsumexp(logits, dim=-1),
                atol=1e-5,
                rtol=1e-5,
            )


# ---------------------------------------------------------------------------
# GPU tests
# ---------------------------------------------------------------------------


def _gpu_skip_reason():
    if not torch.cuda.is_available():
        return "CUDA required"
    if torch.cuda.get_device_capability() not in SUPPORTED_COMPUTE_CAPABILITIES:
        return "SM100 or SM103 required"
    if not cake_backend.generated_program_available(torch.device("cuda")):
        return "generated programs not registered for this device"
    return None


def test_host_tables_image_matches_the_workspace_regions():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required (pinned host memory)")
    batch, q_len, heads = 4, 6, 64
    kv_lens = [8192] * batch
    plan = build_work_plan(kv_lens, num_heads=heads, num_sms=16, q_len=q_len)
    layout = workspace_layout(plan, batch=batch, num_heads=heads, q_len=q_len)
    q_indptr = _q_indptr(batch, q_len)
    row_splits = token_splits(
        plan, q_indptr, q_len=q_len, num_heads=heads, total_q=batch * q_len
    )
    image = host_tables(plan, layout, q_indptr=q_indptr, row_splits=row_splits)
    assert image.is_pinned() and image.dtype == torch.int32
    base = layout["work_table"][0]
    assert image.numel() * 4 == layout["q_indptr"][0] + layout["q_indptr"][1] - base

    def region(name):
        offset, nbytes = layout[name]
        return image[(offset - base) // 4 : (offset - base + nbytes) // 4]

    assert torch.equal(region("work_table"), work_table_rows(plan).reshape(-1))
    assert region("unit_first").tolist() == list(plan.unit_first)
    assert region("row_splits").tolist() == row_splits
    assert region("q_indptr").tolist() == q_indptr


def test_rejects_unsupported_compute_capability():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability() in SUPPORTED_COMPUTE_CAPABILITIES:
        pytest.skip("device is supported; nothing to reject")
    inputs = make_inputs([64], 128, q_len=1, enable_sink=False, device="cuda")
    workspace = torch.empty(1 << 20, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="compute capability"):
        prepare_nvfp4_batch_decode_with_kv_cache_mla(
            inputs["query"],
            inputs["query_scale"],
            inputs["kv_cache"],
            inputs["kv_scale"],
            inputs["block_tables"],
            inputs["seq_lens"],
            workspace,
            sm_scale=inputs["sm_scale"],
        )


@pytest.mark.parametrize(
    "label,kv_lens,q_len,num_heads,enable_sink,tiles_per_split",
    [
        ("smoke_forced_two_splits", [256, 300], 6, 64, True, 2),
        ("partial_pages_h8", [1000, 8192, 5000], 6, 8, True, None),
        ("no_sink_q1", [64, 65, 4096, 777], 1, 64, False, None),
        ("bs32_q6_kv8k", [8192] * 32, 6, 64, True, None),
        ("bs2_kv16k_mixed", [16384, 12000], 6, 64, False, None),
    ],
)
def test_nvfp4_mla_decode(
    label, kv_lens, q_len, num_heads, enable_sink, tiles_per_split
):
    reason = _gpu_skip_reason()
    if reason:
        pytest.skip(reason)
    inputs = make_inputs(
        kv_lens, num_heads, q_len=q_len, enable_sink=enable_sink, device="cuda"
    )
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    if tiles_per_split is None:
        nbytes = nvfp4_mla_decode_workspace_size(
            kv_lens, num_heads, num_sms=num_sms, q_len=q_len, enable_sink=enable_sink
        )
    else:
        nbytes = max_nvfp4_mla_decode_workspace_size(
            len(kv_lens), num_heads, q_len=q_len
        )
    # Poisoned workspace: prepare does not initialize the partial regions, so
    # every slot the kernels read must be written by them first (0xFF bytes are
    # NaN as BF16 / FP32 and -1 as int32).
    workspace = torch.full((nbytes,), 0xFF, dtype=torch.uint8, device="cuda")
    out = torch.empty(
        (len(kv_lens) * q_len, num_heads, HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    )
    lse = torch.full(
        (len(kv_lens) * q_len, num_heads),
        float("nan"),
        dtype=torch.float32,
        device="cuda",
    )
    decode = cake_backend.prepare_nvfp4_batch_decode_with_kv_cache_mla(
        inputs["query"],
        inputs["query_scale"],
        inputs["kv_cache"],
        inputs["kv_scale"],
        inputs["block_tables"],
        inputs["seq_lens"],
        workspace,
        sm_scale=inputs["sm_scale"],
        sinks=inputs["sinks"],
        out=out,
        lse=lse,
        return_lse=True,
        tiles_per_split=tiles_per_split,
    )
    assert decode.main_kwargs["grid"] == (CLUSTER_PAIR * decode.plan.num_units, 1, 1)
    assert decode.main_kwargs["q_len"] == q_len
    assert_plan_well_formed(
        decode.plan, kv_lens, q_len=q_len, num_heads=num_heads, enable_sink=enable_sink
    )
    total_q = len(kv_lens) * q_len
    q_indptr = _q_indptr(len(kv_lens), q_len)
    torch.cuda.synchronize()  # the single table upload is asynchronous
    assert torch.equal(
        decode.main_kwargs["work_table"].cpu(), work_table_rows(decode.plan)
    )
    assert decode.main_kwargs["unit_first"].cpu().tolist() == list(
        decode.plan.unit_first
    )
    assert decode.main_kwargs["q_indptr"].cpu().tolist() == q_indptr
    if tiles_per_split is not None:
        assert decode.plan.max_splits >= 2 and decode.reduce_kwargs is not None
    if decode.reduce_kwargs is not None:
        expected_splits = token_splits(
            decode.plan,
            q_indptr,
            q_len=q_len,
            num_heads=num_heads,
            total_q=total_q,
        )
        assert decode.reduce_kwargs["row_splits"].cpu().tolist() == expected_splits
        table = decode.main_kwargs["work_table"].cpu()
        assert torch.equal((table[:, 7] & FLAG_DIRECT_OUT) != 0, table[:, 6] == 1)
    result = decode()
    assert result[0] is out and result[1] is lse
    torch.cuda.synchronize()
    ref_out, ref_lse = reference(
        inputs["query"],
        inputs["query_scale"],
        inputs["kv_cache"],
        inputs["kv_scale"],
        inputs["block_tables"],
        kv_lens,
        q_len,
        inputs["sm_scale"],
        inputs["sinks"],
    )
    check_outputs(out, lse, ref_out, ref_lse)
    snapshot_out, snapshot_lse = out.clone(), lse.clone()
    out.zero_()
    lse.fill_(float("nan"))
    decode()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, snapshot_out, atol=0, rtol=0)
    torch.testing.assert_close(lse, snapshot_lse, atol=0, rtol=0)


def test_public_api_returns_out_without_lse():
    reason = _gpu_skip_reason()
    if reason:
        pytest.skip(reason)
    kv_lens = [512, 300]
    inputs = make_inputs(
        kv_lens, 16, q_len=DSV4_Q_LEN, enable_sink=False, device="cuda"
    )
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    workspace = torch.empty(
        nvfp4_mla_decode_workspace_size(kv_lens, 16, num_sms=num_sms, q_len=DSV4_Q_LEN),
        dtype=torch.uint8,
        device="cuda",
    )
    decode = prepare_nvfp4_batch_decode_with_kv_cache_mla(
        inputs["query"],
        inputs["query_scale"],
        inputs["kv_cache"],
        inputs["kv_scale"],
        inputs["block_tables"],
        inputs["seq_lens"],
        workspace,
        sm_scale=inputs["sm_scale"],
        seq_lens_cpu=inputs["seq_lens"].cpu(),
    )
    out = decode()
    assert isinstance(out, torch.Tensor) and out.shape == (
        len(kv_lens) * DSV4_Q_LEN,
        16,
        HEAD_DIM,
    )
    torch.cuda.synchronize()
    ref_out, ref_lse = reference(
        inputs["query"],
        inputs["query_scale"],
        inputs["kv_cache"],
        inputs["kv_scale"],
        inputs["block_tables"],
        kv_lens,
        DSV4_Q_LEN,
        inputs["sm_scale"],
        None,
    )
    check_outputs(out, decode.lse, ref_out, ref_lse)
