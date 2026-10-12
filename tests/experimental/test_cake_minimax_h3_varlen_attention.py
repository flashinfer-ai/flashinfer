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

import pytest
import torch

from flashinfer.experimental.minimax_h3_varlen_attention import cake_backend
from flashinfer.experimental.minimax_h3_varlen_attention.cake_backend import (
    BLOCK_M,
    SPLIT_PROGRAM_COST,
    CLUSTER_Q_ROWS,
    BLOCK_N,
    COMBINE_WORDS,
    MAX_KV_SPLITS,
    PARTIAL_ROWS,
    UNIT_WORDS,
    BF16_UNIT_OVERHEAD_BLOCKS,
    HEAD_DIM,
    QUANTIZE_SUBS_PER_BLOCK,
    SUPPORTED_COMPUTE_CAPABILITIES,
    TileTables,
    assign_unit_slots,
    bf16_choose_q_splits,
    bf16_unit_cost,
    build_bf16_segment_plan,
    lpt_makespan,
    choose_kv_splits,
    split_chunks,
    build_packed_segment_plan,
    build_tile_tables,
    normalize_cu_seqlens,
    nvfp4_quantize_grid,
    nvfp4_workspace_shapes,
    prepare_minimax_h3_varlen_attention,
    prepare_minimax_h3_varlen_nvfp4_attention,
)
from flashinfer.prefill import (
    minimax_h3_varlen_attention,
    minimax_h3_varlen_nvfp4_attention,
)

# Tolerances of the Cake evaluation contracts (never looser).
BF16_ATOL = BF16_RTOL = 1e-2
NVFP4_ATOL, NVFP4_RTOL = 1.0, 0.1
REFERENCE_QUERY_CHUNK = 1024

# Contract rows (cu_seqlens, local heads).  ``H = 56 / P`` for Ulysses degree P.
SMOKE_ROWS = [
    ("smoke_p8_1", [0, 1], 7),
    ("smoke_p8_128", [0, 128], 7),
    ("smoke_p8_129", [0, 129], 7),
    ("smoke_p8_133_300", [0, 133, 300], 7),
    ("smoke_p8_64_64_200", [0, 64, 128, 200], 7),
    ("smoke_p8_511", [0, 511], 7),
    ("smoke_p8_512", [0, 512], 7),
    ("smoke_p8_513", [0, 513], 7),
    ("empty_segments", [0, 0, 640, 640, 1200, 1201], 7),
]
SEGMENT_ROWS = [
    ("seg3_5s_p8", [0, 4310, 4567, 4824], 7),
    ("seg4_6s_p2", [0, 12285, 16401, 20393, 24384], 28),
]
CENTER_ROWS = [
    ("center_4s_p8_m4184", [0, 4184], 7),
    ("center_5s_p4_m9648", [0, 9648], 14),
    ("tail_5s_p8_m4763", [0, 4763], 7),
    ("pad_5s_p4_used9611_total9664", [0, 9611, 9664], 14),
]
LONG_ROWS = [
    ("center_8s_p2_m29472", [0, 29472], 28),
    ("center_10s_p1_m74240", [0, 74240], 56),
]


def _ceil_div(a, b):
    return -(-a // b)


# Persistent-grid capacities in 2-CTA clusters: B200/B300 (148 SMs), a small
# grid that forces multi-round schedules with a partial tail round, and one.
PLAN_GRID_CLUSTERS = [74, 5, 1]


# ---------------------------------------------------------------------------
# Host plans (CPU)
# ---------------------------------------------------------------------------


def test_assign_unit_slots_lpt():
    # costs sorted descending: units 0(5), 4(4), 2(3), 3(2), 1(1); G=2, tail=1.
    # Round 0 gets [0, 4] -> slots (4, 0); round 1 gets [2, 3] -> slots (3, 2);
    # the tail round hands the cheapest unit to cluster 0.
    assert assign_unit_slots([5, 1, 3, 2, 4], 2) == [4, 0, 3, 2, 1]
    assert assign_unit_slots([], 4) == []
    # Fewer units than clusters: one round, descending cost reversed.
    assert assign_unit_slots([1, 3, 2], 8) == [0, 2, 1]
    # Equal costs keep the enumeration order within a round.
    assert assign_unit_slots([2, 2, 2, 2], 2) == [0, 1, 2, 3]


def _decode_unit_table(plan):
    """``(segment, head, cluster, kv_begin, kv_blocks, slot)`` per scheduled unit."""
    table = plan.unit_table.tolist()
    # ``num_clusters`` zero records of padding follow the scheduled units.
    assert len(table) == UNIT_WORDS * (plan.total_tiles + plan.num_clusters)
    assert table[UNIT_WORDS * plan.total_tiles :] == [0] * (
        UNIT_WORDS * plan.num_clusters
    )
    begins = plan.seg_begin.tolist()
    lens = plan.seg_len.tolist()
    decoded = []
    for u in range(plan.total_tiles):
        record = table[UNIT_WORDS * u : UNIT_WORDS * (u + 1)]
        seg = begins.index(record[0])
        assert record[1] == lens[seg] and record[5:] == [0, 0, 0]
        decoded.append(
            (
                seg,
                record[2] >> 16,
                record[2] & 0xFFFF,
                record[3] >> 16,
                record[3] & 0xFFFF,
                record[4],
            )
        )
    return decoded


def _check_split_units(units, combine, blocks_of, num_partial_slots, num_combine_units):
    """Shared K/V-split invariants of both planners.

    ``units`` are ``(key, kv_begin, kv_blocks, slot)`` rows in slot order,
    ``combine`` the flat combine table, ``blocks_of[key]`` the K/V block count
    of the unit's segment.  Every unit key is covered exactly once by its
    ranges; an unsplit unit is one range with slot -1; a split unit's ranges
    are the near-equal chunks in consecutive partial slots, recorded once in the
    combine table.
    """
    by_key = {}
    for key, begin, count, slot in units:
        by_key.setdefault(key, []).append((begin, count, slot))
    assert len(combine) == COMBINE_WORDS * num_combine_units
    combine_rows = {
        (
            combine[COMBINE_WORDS * i],
            combine[COMBINE_WORDS * i + 1] >> 16,
            combine[COMBINE_WORDS * i + 1] & 0xFFFF,
        ): (
            combine[COMBINE_WORDS * i + 2],
            combine[COMBINE_WORDS * i + 3],
        )
        for i in range(num_combine_units)
    }
    assert len(combine_rows) == num_combine_units
    slots_seen = []
    for key, ranges in by_key.items():
        ranges.sort()
        blocks = blocks_of[key]
        assert ranges[0][0] == 0 and ranges[-1][0] + ranges[-1][1] == blocks
        for (b0, n0, _), (b1, _, _) in zip(ranges, ranges[1:], strict=False):
            assert b0 + n0 == b1
        if len(ranges) == 1:
            assert ranges[0][2] == -1 and key not in combine_rows
            continue
        assert 2 <= len(ranges) <= MAX_KV_SPLITS
        assert [(b, n) for b, n, _ in ranges] == split_chunks(blocks, len(ranges))
        first, splits = combine_rows[key]
        assert splits == len(ranges)
        assert [slot for _, _, slot in ranges] == list(range(first, first + splits))
        slots_seen.extend(range(first, first + splits))
    assert set(combine_rows) <= set(by_key)
    assert sorted(slots_seen) == list(range(num_partial_slots))


@pytest.mark.parametrize("kv_splits", [None, 1, 3])
@pytest.mark.parametrize("grid_clusters", PLAN_GRID_CLUSTERS)
@pytest.mark.parametrize("label,cu,heads", SMOKE_ROWS + SEGMENT_ROWS + CENTER_ROWS)
def test_bf16_segment_plan(label, cu, heads, grid_clusters, kv_splits):
    plan = build_bf16_segment_plan(
        cu,
        torch.device("cpu"),
        heads,
        num_clusters=grid_clusters,
        kv_splits=kv_splits,
        # The plain enumeration: the tail-round Q-split has its own tests below.
        q_split=False,
    )
    lengths = [b - a for a, b in zip(cu, cu[1:], strict=False) if b > a]
    clusters = [_ceil_div(n, CLUSTER_Q_ROWS) for n in lengths]
    blocks = [_ceil_div(n, BLOCK_N) for n in lengths]
    assert plan.num_heads == heads
    assert plan.num_segments == len(lengths)
    assert plan.total_clusters == sum(clusters)
    assert plan.num_clusters == min(grid_clusters, max(plan.total_tiles, 1))
    assert (
        plan.seg_begin.dtype
        == plan.seg_len.dtype
        == plan.unit_table.dtype
        == plan.combine_table.dtype
        == torch.int32
    )
    assert (
        plan.partial_O.dtype == torch.float16 and plan.partial_ML.dtype == torch.float32
    )
    assert (
        plan.partial_O.numel()
        == max(plan.num_partial_slots, 1) * PARTIAL_ROWS * HEAD_DIM
    )
    assert plan.partial_ML.numel() == max(plan.num_partial_slots, 1) * PARTIAL_ROWS * 2
    assert plan.seg_len.tolist() == lengths
    assert plan.seg_begin.tolist() == [
        a for a, b in zip(cu, cu[1:], strict=False) if b > a
    ]
    decoded = _decode_unit_table(plan)
    expected = [
        (s, h, c)
        for s, n in enumerate(clusters)
        for h in range(heads)
        for c in range(n)
    ]
    # Every (segment, head, cluster) unit is covered exactly once by its K/V
    # ranges; split units have consecutive partial slots and one combine row.
    assert sorted({(s, h, c) for s, h, c, *_ in decoded}) == expected
    _check_split_units(
        [((s, h, c), b, n, slot) for s, h, c, b, n, slot in decoded],
        plan.combine_table.tolist() if plan.num_combine_units else [],
        {(s, h, c): blocks[s] for s, h, c in expected},
        plan.num_partial_slots,
        plan.num_combine_units,
    )
    # The split factors are the planner's (forced or chosen) per-unit factors.
    split_of = choose_kv_splits(
        [blocks[s] for s, _h, _c in expected], plan.num_clusters, force=kv_splits
    )
    assert plan.max_kv_splits == max(split_of, default=1)
    assert plan.total_tiles == sum(
        min(k, blocks[s]) for k, (s, _h, _c) in zip(split_of, expected, strict=True)
    )
    if kv_splits == 1:
        assert plan.total_tiles == heads * plan.total_clusters
        assert plan.num_combine_units == 0 and plan.num_partial_slots == 0
    # Slot order is the longest-processing-time-first assignment of the
    # segment-major enumeration (ranges in K/V order) under the per-range cost.
    ranges = [
        ((s, h, c), b, n)
        for k, (s, h, c) in zip(split_of, expected, strict=True)
        for b, n in split_chunks(blocks[s], k)
    ]
    # Units with at most 256 valid Q rows run one Q stage at a discounted cost.
    cost = [
        bf16_unit_cost(n, lengths[s] - c * CLUSTER_Q_ROWS <= CLUSTER_Q_ROWS // 2)
        for (s, _h, c), _b, n in ranges
    ]
    slots = assign_unit_slots(cost, plan.num_clusters)
    assert [((s, h, c), b, n) for s, h, c, b, n, _ in decoded] == [
        ranges[u] for u in slots
    ]
    G = plan.num_clusters
    for k in range(plan.total_tiles // G):
        round_costs = [cost[slots[k * G + i]] for i in range(G)]
        assert round_costs == sorted(round_costs)


def test_choose_kv_splits_policy():
    # Four or more waves: never split.
    assert choose_kv_splits([40] * 400, 74) == [1] * 400
    # One unit per cluster and a partial wave of expensive units: the tail is
    # split so the makespan drops below the unsplit wave.
    blocks = [200] * 80
    split_of = choose_kv_splits(blocks, 74)
    assert len(split_of) == 80 and max(split_of) >= 2 and min(split_of) >= 1
    assert all(1 <= k <= MAX_KV_SPLITS for k in split_of)
    # Cheap units gain nothing from splitting (combine cost dominates).
    assert choose_kv_splits([2] * 80, 74) == [1] * 80
    # Forced factors are clamped to the unit's block count.
    assert choose_kv_splits([1, 5, 9], 74, force=4) == [1, 4, 4]
    # A split-capable program that costs 1.21x per unit: the 1.14-wave row still
    # splits (the wave-quantization gain exceeds the program cost), a 3.2-wave
    # row no longer does.
    assert choose_kv_splits([48] * 84, 74, program_cost=1.21) != [1] * 84
    assert choose_kv_splits([66] * 238, 74, program_cost=1.0) != [1] * 238
    assert choose_kv_splits([66] * 238, 74, program_cost=1.21) == [1] * 238
    # The production sm_103a fp8pv cost (1.01: the split program runs at parity
    # with the dense program) splits both the 1.42-wave row (105 units of 58
    # blocks) and the 1.14-wave row (84 units of 48).
    cost = SPLIT_PROGRAM_COST[("fp8", "sm_103a")]
    assert cost <= 1.05
    assert choose_kv_splits([58] * 105, 74, program_cost=cost) != [1] * 105
    assert choose_kv_splits([48] * 84, 74, program_cost=cost) != [1] * 84
    assert split_chunks(10, 4) == [(0, 3), (3, 3), (6, 2), (8, 2)]
    assert split_chunks(2, 8) == [(0, 1), (1, 1)]


# center_5s_p2_m19296 / center_10s_p2_m37120 of the Cake BF16 contract: 1064 /
# 2044 units on the 74-cluster grid.
Q_SPLIT_CENTER_5S_P2 = ([0, 19296], 28)
Q_SPLIT_CENTER_10S_P2 = ([0, 37120], 28)
HALF_UNIT_ROWS = CLUSTER_Q_ROWS // 2


def _bf16_records(plan):
    """The ``total_tiles`` eight-word unit records of a plan; the ``num_clusters`` padding records are zero."""
    words = plan.unit_table.tolist()
    total = int(plan.total_tiles)
    assert len(words) == UNIT_WORDS * (total + int(plan.num_clusters))
    assert all(word == 0 for word in words[UNIT_WORDS * total :])
    return [
        tuple(words[i : i + UNIT_WORDS])
        for i in range(0, UNIT_WORDS * total, UNIT_WORDS)
    ]


def _decode_q_split_record(record):
    """The kernel's decode of record word 5: ``(head, cluster, unit_row0, two_stages, slot, q_half)``."""
    _doc_begin, doc_len, packed, _kv_words, slot, q_half, zero_6, zero_7 = record
    assert (zero_6, zero_7) == (0, 0)
    head, c = packed >> 16, packed & 0xFFFF
    unit_row0 = c * CLUSTER_Q_ROWS + (q_half >> 1) * HALF_UNIT_ROWS
    full_rows = int(doc_len - unit_row0 > HALF_UNIT_ROWS)
    two_stages = full_rows if q_half == 0 else 0
    return head, c, unit_row0, two_stages, slot, q_half


def _assert_q_split_cover(records, cu, num_heads):
    """Every segment row of every head is computed and stored by exactly one record."""
    spans = {}
    for record in records:
        doc_begin, doc_len = record[0], record[1]
        head, _c, unit_row0, two_stages, _slot, _q = _decode_q_split_record(record)
        rows = CLUSTER_Q_ROWS if two_stages else HALF_UNIT_ROWS
        spans.setdefault((doc_begin, head), []).append(
            (unit_row0, min(unit_row0 + rows, doc_len))
        )
    segments = {a: b - a for a, b in zip(cu, cu[1:], strict=False) if b > a}
    assert set(spans) == {
        (begin, head) for begin in segments for head in range(num_heads)
    }
    for (begin, _head), ranges in spans.items():
        cursor = 0
        for lo, hi in sorted(ranges):
            assert lo == cursor and hi > lo, (begin, sorted(ranges))
            cursor = hi
        assert cursor == segments[begin], (begin, sorted(ranges))


def test_bf16_choose_q_splits_policy():
    # center_5s_p2 on 74 clusters: 1064 two-stage units of 151 K/V blocks, 28 in
    # the tail round, 46 idle clusters: the whole tail round is halved.
    full, half = bf16_unit_cost(151, False), bf16_unit_cost(151, True)
    assert (full, half) == (153, 93)
    costs = [full] * 1064
    flags = bf16_choose_q_splits(costs, [half] * 1064, 74)
    assert sum(flags) == 28 and all(flags[1036:]) and not any(flags[:1036])
    assert lpt_makespan(costs, 74) == 15 * full == 2295
    assert lpt_makespan(costs[:1036] + [half] * 56, 74) == 14 * full + half == 2235
    # The smallest paying prefix of the most expensive tail units.  Tail round:
    # 14 x 153, 10 x 100, 4 x 50; halving the 153s alone leaves the 100s as the
    # tail maximum, halving the 100s too pays, the 50s change nothing.
    costs = [153] * 1050 + [100] * 10 + [50] * 4
    halves = [93] * 1050 + [62] * 10 + [32] * 4
    flags = bf16_choose_q_splits(costs, halves, 74)
    assert [u for u, flag in enumerate(flags) if flag] == list(range(1036, 1060))
    sim = costs[:1036] + [93] * 28 + [62] * 20 + [50] * 4
    assert lpt_makespan(costs, 74) == 2295 and lpt_makespan(sim, 74) == 2235
    # Single-stage units (no half cost: nothing to halve) and K/V-split ranges
    # are never halved.  center_10s_p2: 73 clusters per head, the 73rd a 256-row
    # single-stage unit (cost 176, one half's cost); 46 tail units (18 two-stage
    # + 28 single-stage) on 28 idle clusters: the two-stage ones are halved.
    assert (bf16_unit_cost(290, False), bf16_unit_cost(290, True)) == (292, 176)
    costs = [bf16_unit_cost(290, c == 72) for _head in range(28) for c in range(73)]
    halves = [
        bf16_unit_cost(290, True) if c < 72 else None
        for _head in range(28)
        for c in range(73)
    ]
    flags = bf16_choose_q_splits(costs, halves, 74)
    assert [u for u, flag in enumerate(flags) if flag] == [
        27 * 73 + c for c in range(54, 72)
    ]
    sim = [292] * 1998 + [176] * 64
    assert lpt_makespan(costs, 74) == 28 * 292
    assert lpt_makespan(sim, 74) == 27 * 292 + 176
    assert not any(bf16_choose_q_splits([153] * 1064, [None] * 1064, 74))
    # No tail round (nothing idles); 46 uniform tail units on 28 idle clusters
    # (an unhalved tail unit keeps the makespan); sub-wave plans: 37 units fit
    # as 74 halves, 70 units leave 4 idle clusters only.
    assert not any(bf16_choose_q_splits([], [], 74))
    assert not any(bf16_choose_q_splits([153] * 148, [93] * 148, 74))
    assert not any(bf16_choose_q_splits([153] * 74, [93] * 74, 74))
    assert not any(bf16_choose_q_splits([153] * 120, [93] * 120, 74))
    assert all(bf16_choose_q_splits([153] * 37, [93] * 37, 74))
    assert not any(bf16_choose_q_splits([153] * 70, [93] * 70, 74))
    # The makespan is set by clusters without a tail unit (round 5 gives them
    # 153 while the tail-owning clusters got 42): halving the cheap tail units
    # cannot shorten it.
    costs, halves = [153] * 304 + [42] * 80, [93] * 304 + [26] * 80
    assert not any(bf16_choose_q_splits(costs, halves, 74))


def test_bf16_choose_q_splits_min_gain(monkeypatch):
    # 81 waves: halving the 37 tail units saves 60 of 12393 (0.48 %), below the 0.5 % bar.
    costs, halves = [153] * (74 * 80 + 37), [93] * (74 * 80 + 37)
    assert cake_backend.BF16_Q_SPLIT_MIN_GAIN == 0.005
    assert not any(bf16_choose_q_splits(costs, halves, 74))
    monkeypatch.setattr(cake_backend, "BF16_Q_SPLIT_MIN_GAIN", 0.0)
    flags = bf16_choose_q_splits(costs, halves, 74)
    assert sum(flags) == 37 and all(flags[74 * 80 :])
    # 61 waves: the same halving saves 60 of 9333 (0.64 %): whole at the 1 % bar, halved at 0.5 %.
    costs, halves = [153] * (74 * 60 + 37), [93] * (74 * 60 + 37)
    monkeypatch.setattr(cake_backend, "BF16_Q_SPLIT_MIN_GAIN", 0.01)
    assert not any(bf16_choose_q_splits(costs, halves, 74))
    monkeypatch.setattr(cake_backend, "BF16_Q_SPLIT_MIN_GAIN", 0.005)
    flags = bf16_choose_q_splits(costs, halves, 74)
    assert sum(flags) == 37 and all(flags[74 * 60 :])


def test_bf16_q_split_plan_center_5s_p2():
    cu, heads = Q_SPLIT_CENTER_5S_P2
    cpu = torch.device("cpu")
    plain = build_bf16_segment_plan(cu, cpu, heads, num_clusters=74, q_split=False)
    assert plain.num_q_split_units == 0 and plain.total_tiles == 1064
    assert plain.num_clusters == 74
    records = _bf16_records(plain)
    # Uniform costs reproduce the plain enumeration (slot k runs unit k, heads
    # slow); word 5 and the spare words are 0.
    assert records == [
        (0, 19296, ((k // 38) << 16) | (k % 38), 151, -1, 0, 0, 0) for k in range(1064)
    ]
    _assert_q_split_cover(records, cu, heads)
    # The default (``BF16_Q_SPLIT_PLAN``) halves the 28 tail-round units: the 14
    # full rounds are the plain plan's, the tail round holds the 56 halves, each
    # pair on neighbouring slots.
    assert cake_backend.BF16_Q_SPLIT_PLAN is True
    plan = build_bf16_segment_plan(cu, cpu, heads, num_clusters=74)
    assert plan.num_q_split_units == 28 and plan.total_tiles == 1064 + 28
    assert plan.num_clusters == 74
    assert plan.num_partial_slots == 0 and plan.num_combine_units == 0
    explicit = build_bf16_segment_plan(cu, cpu, heads, num_clusters=74, q_split=True)
    assert explicit.unit_table.tolist() == plan.unit_table.tolist()
    split_records = _bf16_records(plan)
    full_rounds = 14 * 74
    assert split_records[:full_rounds] == records[:full_rounds]
    tail = [_decode_q_split_record(record) for record in split_records[full_rounds:]]
    assert [(head, c, q) for head, c, _u0, _ts, _slot, q in tail] == [
        (27, c, q) for c in range(10, 38) for q in (1, 2)
    ]
    for _head, c, unit_row0, two_stages, slot, q in tail:
        assert unit_row0 == c * 512 + (q - 1) * 256 and two_stages == 0 and slot == -1
    # Whole records decode as before: two stages whenever more than 256 rows remain.
    for record in split_records[:full_rounds]:
        _head, c, unit_row0, two_stages, _slot, q = _decode_q_split_record(record)
        assert q == 0 and unit_row0 == c * 512
        assert two_stages == int(19296 - unit_row0 > 256)
    _assert_q_split_cover(split_records, cu, heads)


def test_bf16_q_split_plan_center_10s_p2():
    cu, heads = Q_SPLIT_CENTER_10S_P2
    plan = build_bf16_segment_plan(
        cu, torch.device("cpu"), heads, num_clusters=74, q_split=True
    )
    assert plan.num_q_split_units == 18 and plan.total_tiles == 2044 + 18
    assert plan.num_clusters == 74
    records = _bf16_records(plan)
    tail = [_decode_q_split_record(record) for record in records[27 * 74 :]]
    assert len(tail) == 64
    # Stable LPT order on the equal costs: the single-stage last-cluster units of
    # heads 0..26, the 36 halves of head 27's units 54..71, then head 27's own
    # single-stage unit.
    expected = (
        [(h, 72, 0) for h in range(27)]
        + [(27, c, q) for c in range(54, 72) for q in (1, 2)]
        + [(27, 72, 0)]
    )
    assert [(head, c, q) for head, c, _u0, _ts, _slot, q in tail] == expected
    assert all(two_stages == 0 for _h, _c, _u0, two_stages, _s, _q in tail)
    _assert_q_split_cover(records, cu, heads)


def test_bf16_q_split_plan_reads_the_module_option(monkeypatch):
    cu, heads = Q_SPLIT_CENTER_5S_P2
    cpu = torch.device("cpu")
    monkeypatch.setattr(cake_backend, "BF16_Q_SPLIT_PLAN", False)
    plan = build_bf16_segment_plan(cu, cpu, heads, num_clusters=74)
    assert plan.num_q_split_units == 0 and plan.total_tiles == 1064
    forced = build_bf16_segment_plan(cu, cpu, heads, num_clusters=74, q_split=True)
    assert forced.num_q_split_units == 28 and forced.total_tiles == 1064 + 28


def test_bf16_q_split_plan_leaves_full_waves_and_kv_split_plans_alone():
    cpu = torch.device("cpu")
    # 28 heads x 37 clusters = 1036 = 14 x 74 units: no tail round, nothing idles.
    cu, heads = [0, 37 * CLUSTER_Q_ROWS], 28
    plan = build_bf16_segment_plan(cu, cpu, heads, num_clusters=74)
    plain = build_bf16_segment_plan(cu, cpu, heads, num_clusters=74, q_split=False)
    assert plan.num_q_split_units == 0 and plan.total_tiles == 1036
    assert plan.unit_table.tolist() == plain.unit_table.tolist()
    # K/V-split ranges are never halved: a forced four-way split plan is
    # unchanged by the option.
    cu, heads = [0, 133, 300, 900, 3000], 7
    base = build_bf16_segment_plan(
        cu, cpu, heads, num_clusters=74, kv_splits=4, q_split=False
    )
    plan = build_bf16_segment_plan(
        cu, cpu, heads, num_clusters=74, kv_splits=4, q_split=True
    )
    assert base.num_partial_slots > 0 and plan.num_q_split_units == 0
    assert plan.unit_table.tolist() == base.unit_table.tolist()
    assert plan.combine_table.tolist() == base.combine_table.tolist()


@pytest.mark.parametrize("label,cu,heads", SMOKE_ROWS + SEGMENT_ROWS + CENTER_ROWS)
def test_packed_segment_plan(label, cu, heads):
    plan = build_packed_segment_plan(cu, torch.device("cpu"))
    lengths = [b - a for a, b in zip(cu, cu[1:], strict=False) if b > a]
    assert plan.seg_len == tuple(lengths)
    assert sum(_ceil_div(n, BLOCK_M) for n in lengths) == plan.PB
    assert plan.total_clusters == sum(_ceil_div(n, CLUSTER_Q_ROWS) for n in lengths)
    if not lengths:
        return
    # Every THD token maps to exactly one packed row; padded rows are marked.
    block_token = plan.block_token.tolist()
    block_valid = plan.block_valid.tolist()
    assert len(block_token) == len(block_valid) == plan.PB
    covered = []
    for token, valid in zip(block_token, block_valid, strict=True):
        assert 0 <= valid <= BLOCK_M
        covered.extend(range(token, token + valid))
    assert covered == [t for a, b in zip(cu, cu[1:], strict=False) for t in range(a, b)]
    for s in range(plan.num_segments):
        first = plan.seg_tile_base[s]
        assert block_token[first] == plan.seg_begin[s]
        assert sum(block_valid[first : first + plan.seg_blocks[s]]) == plan.seg_len[s]
    # Cluster tables: each cluster tile covers four Q blocks of one segment.
    cl_q_block = plan.cl_q_block.tolist()
    cl_seg_len = plan.cl_seg_len.tolist()
    cl_kv_base = plan.cl_kv_base.tolist()
    assert len(cl_q_block) == plan.total_clusters
    for s in range(plan.num_segments):
        lo, hi = plan.cluster_off[s], plan.cluster_off[s + 1]
        assert cl_q_block[lo:hi] == [4 * c for c in range(hi - lo)]
        assert set(cl_seg_len[lo:hi]) == {plan.seg_len[s]}
        assert set(cl_kv_base[lo:hi]) == {plan.seg_tile_base[s]}


@pytest.mark.parametrize("kv_splits", [None, 1, 3])
@pytest.mark.parametrize("grid_clusters", PLAN_GRID_CLUSTERS)
@pytest.mark.parametrize("heads", [7, 14, 28])
@pytest.mark.parametrize(
    "label,cu,heads_unused", SMOKE_ROWS + SEGMENT_ROWS + CENTER_ROWS
)
def test_nvfp4_tile_tables(label, cu, heads_unused, heads, grid_clusters, kv_splits):
    plan = build_packed_segment_plan(cu, torch.device("cpu"))
    tiles = build_tile_tables(
        plan,
        heads,
        torch.device("cpu"),
        num_clusters=grid_clusters,
        kv_splits=kv_splits,
    )
    assert tiles.heads == heads
    tables = {name: getattr(tiles, name) for name in TileTables.NAMES}
    for name, table in tables.items():
        assert table.dtype == torch.int32, name
        assert table.shape == (max(tiles.total_tiles, 1),), name
    assert tiles.seg_begin.tolist() == list(plan.seg_begin)
    assert tiles.seg_len.tolist() == list(plan.seg_len)
    assert (
        tiles.partial_O.dtype == torch.float16
        and tiles.partial_ML.dtype == torch.float32
    )
    assert (
        tiles.partial_O.numel()
        == max(tiles.num_partial_slots, 1) * PARTIAL_ROWS * HEAD_DIM
    )
    if tiles.total_tiles == 0:
        assert tiles.num_combine_units == 0
        return
    rows = list(zip(*(tables[name].tolist() for name in TileTables.NAMES), strict=True))
    cluster_rows = list(
        zip(
            plan.cl_seg_begin.tolist(),
            plan.cl_seg_len.tolist(),
            plan.cl_kv_base.tolist(),
            plan.cl_q_block.tolist(),
            strict=True,
        )
    )
    segment_of_cluster = [
        s
        for s in range(plan.num_segments)
        for _ in range(plan.cluster_off[s], plan.cluster_off[s + 1])
    ]
    # Every (head, cluster tile) pair is covered exactly once by its K/V ranges
    # (the cluster's own per-cluster entries); split units park in
    # consecutive partial slots with one combine row each.
    assert sorted({row[:5] for row in rows}) == sorted(
        (h, *cluster_rows[c]) for h in range(heads) for c in range(plan.total_clusters)
    )
    combine = tiles.combine_table.tolist() if tiles.num_combine_units else []
    key_of_row = {}
    for c, crow in enumerate(cluster_rows):
        s = segment_of_cluster[c]
        for h in range(heads):
            key_of_row[(h, *crow)] = (s, h, c - plan.cluster_off[s])
    _check_split_units(
        [(key_of_row[row[:5]], row[5], row[6], row[7]) for row in rows],
        combine,
        {key: _ceil_div(plan.seg_len[key[0]], BLOCK_N) for key in key_of_row.values()},
        tiles.num_partial_slots,
        tiles.num_combine_units,
    )
    # Enumeration: segments by descending length (ties in segment order), then
    # head, then the segment's cluster tiles, then the unit's K/V ranges; the
    # slots are that enumeration's longest-processing-time-first assignment.
    segments = sorted(range(plan.num_segments), key=lambda s: -plan.seg_len[s])
    keys = [
        (h, c)
        for s in segments
        for h in range(heads)
        for c in range(plan.cluster_off[s], plan.cluster_off[s + 1])
    ]
    blocks = [_ceil_div(plan.seg_len[segment_of_cluster[c]], BLOCK_N) for _h, c in keys]
    split_of = choose_kv_splits(blocks, grid_clusters, force=kv_splits)
    assert tiles.max_kv_splits == max(split_of)
    ranges = [
        (h, *cluster_rows[c], b, n)
        for k, (h, c), nb in zip(split_of, keys, blocks, strict=True)
        for b, n in split_chunks(nb, k)
    ]
    assert tiles.total_tiles == len(ranges)
    if kv_splits == 1:
        assert tiles.total_tiles == heads * plan.total_clusters
        assert tiles.num_combine_units == 0
    cost = [r[-1] + BF16_UNIT_OVERHEAD_BLOCKS for r in ranges]
    G = min(grid_clusters, tiles.total_tiles)
    slots = assign_unit_slots(cost, G)
    assert [row[:7] for row in rows] == [ranges[u] for u in slots]
    for k in range(tiles.total_tiles // G):
        round_costs = [rows[k * G + i][6] for i in range(G)]
        assert round_costs == sorted(round_costs)


def test_nvfp4_tile_tables_lpt_order():
    # One cluster: the slots are the pure longest-processing-time order -- the
    # 600-token segment's four units (5 + 2 blocks) precede the 167- and
    # 133-token units (2 + 2 blocks, enumeration order kept on ties).
    plan = build_packed_segment_plan([0, 133, 300, 900], torch.device("cpu"))
    tiles = build_tile_tables(plan, 2, torch.device("cpu"), num_clusters=1, kv_splits=1)
    assert tiles.total_tiles == 2 * 4
    assert tiles.cl_seg_len.tolist() == [600, 600, 600, 600, 167, 167, 133, 133]
    assert tiles.cl_head.tolist() == [0, 0, 1, 1, 0, 1, 0, 1]
    assert tiles.cl_q_block.tolist() == [0, 4, 0, 4, 0, 0, 0, 0]
    assert tiles.cl_kv_base.tolist() == [4, 4, 4, 4, 2, 2, 0, 0]
    assert tiles.cl_seg_begin.tolist() == [300, 300, 300, 300, 133, 133, 0, 0]
    assert tiles.cl_kv_begin.tolist() == [0] * 8
    assert tiles.cl_kv_blocks.tolist() == [5, 5, 5, 5, 2, 2, 2, 2]
    assert tiles.cl_ws_slot.tolist() == [-1] * 8
    # Forced two-way split of every unit: the 600-token units become (0, 3) +
    # (3, 2) block ranges in partial slots 0..7, the short units (0, 1) + (1, 1).
    tiles = build_tile_tables(plan, 2, torch.device("cpu"), num_clusters=1, kv_splits=2)
    assert tiles.total_tiles == 16 and tiles.num_combine_units == 8
    assert tiles.num_partial_slots == 16 and tiles.max_kv_splits == 2
    assert tiles.cl_kv_blocks.tolist()[:8] == [3, 3, 3, 3, 2, 2, 2, 2]
    assert sorted(tiles.cl_ws_slot.tolist()) == list(range(16))


def test_nvfp4_quantize_grid():
    assert QUANTIZE_SUBS_PER_BLOCK == 4
    assert nvfp4_quantize_grid(7, 9) == (7 * 9 * 4, 1, 1)
    assert nvfp4_quantize_grid(28, 0) == (0, 1, 1)  # never launched (no tiles)


def test_nvfp4_workspace_shapes():
    shapes = nvfp4_workspace_shapes(7, 3, "fp4")
    assert shapes["q_fp4"] == ((7 * 3 * 128, 64), torch.uint8)
    assert shapes["q_scale"] == ((7 * 3 * 32, 32), torch.uint8)
    assert shapes["v_fp4_t"] == ((7 * 128, 3 * 64), torch.uint8)
    assert (
        shapes["v_scale_lo"] == shapes["v_scale_hi"] == ((7 * 3 * 16, 32), torch.uint8)
    )
    shapes = nvfp4_workspace_shapes(7, 0, "fp8")  # PB padded to one block
    assert shapes["v_fp8"] == ((7 * 128, 128), torch.uint8)
    assert shapes["v_amax_partial"] == ((cake_backend.AMAX_CTAS,), torch.float32)
    assert shapes["v_amax"] == ((1,), torch.float32)
    with pytest.raises(ValueError, match="pv_mode"):
        nvfp4_workspace_shapes(7, 3, "bf16")


@pytest.mark.parametrize("cu", [[1, 2], [0, 5, 3], [0], [0, 3, 3, 2]])
def test_rejects_invalid_cu_seqlens(cu):
    with pytest.raises(ValueError):
        normalize_cu_seqlens(cu)


def test_rejects_total_mismatch():
    with pytest.raises(ValueError, match="token extent"):
        normalize_cu_seqlens([0, 10], total_tokens=11)


def test_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        minimax_h3_varlen_attention(None, None, None, None, backend="unknown")
    with pytest.raises(ValueError, match="backend"):
        minimax_h3_varlen_nvfp4_attention(None, None, None, None, backend="unknown")


# ---------------------------------------------------------------------------
# GPU correctness
# ---------------------------------------------------------------------------


def _arch():
    if not torch.cuda.is_available():
        return None
    return SUPPORTED_COMPUTE_CAPABILITIES.get(torch.cuda.get_device_capability(0))


def _require_program(variant):
    arch = _arch()
    if arch is None:
        pytest.skip("MiniMax-H3 varlen attention requires an SM100/SM103 GPU")
    if not cake_backend.route_available(variant, arch):
        pytest.skip(f"no generated {variant} program registered for {arch}")
    return arch


def _make_inputs(cu, heads, seed, device="cuda"):
    gen = torch.Generator(device=device).manual_seed(seed)
    shape = (cu[-1], heads, HEAD_DIM)
    q = torch.randn(shape, dtype=torch.bfloat16, device=device, generator=gen)
    k = torch.randn(shape, dtype=torch.bfloat16, device=device, generator=gen)
    v = torch.randn(shape, dtype=torch.bfloat16, device=device, generator=gen)
    cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=device)
    return q, k, v, cu_seqlens


def _make_engine_views(cu, heads, seed, layout, device="cuda"):
    """Q/K/V as the engine's strided ``[T, H, 128]`` views (never copied).

    ``"fused_qkv"``: column chunks of the fused QKV projection ``[T, 3 * H * 128]``
    (strides ``(3 * H * 128, 128, 1)``).  ``"pack"``: the kind slices of the Cake
    pre-attention pack ``[T, H, 3, 128]`` (strides ``(H * 384, 384, 1)``).
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    total = cu[-1]
    if layout == "fused_qkv":
        qkv = torch.randn(
            (total, 3 * heads * HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
            generator=gen,
        )
        width = heads * HEAD_DIM
        q, k, v = (
            qkv[:, i * width : (i + 1) * width].view(total, heads, HEAD_DIM)
            for i in range(3)
        )
        expected_strides = (3 * heads * HEAD_DIM, HEAD_DIM, 1)
    elif layout == "pack":
        pack = torch.randn(
            (total, heads, 3, HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
            generator=gen,
        )
        q, k, v = (pack[:, :, i, :] for i in range(3))
        expected_strides = (heads * 3 * HEAD_DIM, 3 * HEAD_DIM, 1)
    else:
        raise ValueError(layout)
    for t in (q, k, v):
        assert tuple(t.shape) == (total, heads, HEAD_DIM)
        assert t.stride() == expected_strides
        assert total == 0 or not t.is_contiguous()
    cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=device)
    return q, k, v, cu_seqlens


def _reference(q, k, v, cu, scale):
    """Per-segment, per-head FP32 oracle with TF32 disabled (chunked rows)."""
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        out = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
        for a, b in zip(cu, cu[1:], strict=False):
            if b <= a:
                continue
            for h in range(q.shape[1]):
                keys = k[a:b, h].float()
                values = v[a:b, h].float()
                for r0 in range(a, b, REFERENCE_QUERY_CHUNK):
                    r1 = min(r0 + REFERENCE_QUERY_CHUNK, b)
                    logits = (
                        torch.matmul(q[r0:r1, h].float(), keys.transpose(0, 1)) * scale
                    )
                    out[r0:r1, h] = torch.matmul(torch.softmax(logits, dim=-1), values)
        return out
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def _check(out, expected, atol, rtol):
    assert out.dtype == torch.bfloat16
    assert not torch.isnan(out).any()
    torch.testing.assert_close(out.float(), expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("label,cu,heads", SMOKE_ROWS + SEGMENT_ROWS + CENTER_ROWS)
def test_bf16_matches_fp32_reference(label, cu, heads):
    _require_program("bf16")
    q, k, v, cu_seqlens = _make_inputs(cu, heads, seed=6090)
    scale = 1.0 / math.sqrt(HEAD_DIM)
    out = minimax_h3_varlen_attention(q, k, v, cu_seqlens)
    torch.cuda.synchronize()
    expected = _reference(q, k, v, cu, scale)
    _check(out, expected, BF16_ATOL, BF16_RTOL)
    # Idempotent: the same call reproduces the output bit-exactly.
    again = minimax_h3_varlen_attention(q, k, v, cu_seqlens)
    torch.testing.assert_close(again, out, atol=0, rtol=0)


@pytest.mark.parametrize("label,cu,heads", LONG_ROWS)
def test_bf16_long_segments(label, cu, heads):
    _require_program("bf16")
    q, k, v, cu_seqlens = _make_inputs(cu, heads, seed=6091)
    out = torch.full(q.shape, float("nan"), dtype=torch.bfloat16, device="cuda")
    result = minimax_h3_varlen_attention(
        q, k, v, cu_seqlens, out=out, cu_seqlens_host=cu
    )
    assert result is out
    torch.cuda.synchronize()
    _check(
        out, _reference(q, k, v, cu, 1.0 / math.sqrt(HEAD_DIM)), BF16_ATOL, BF16_RTOL
    )


def test_bf16_prepared_runner_and_graph_replay():
    _require_program("bf16")
    cu, heads = [0, 133, 300, 900], 7
    q, k, v, cu_seqlens = _make_inputs(cu, heads, seed=1)
    out = torch.empty_like(q)
    runner = prepare_minimax_h3_varlen_attention(
        q, k, v, cu_seqlens, out=out, cu_seqlens_host=cu
    )
    assert runner.plan.num_segments == 3 and runner.plan.total_clusters == 4
    assert runner() is out
    torch.cuda.synchronize()
    expected = _reference(q, k, v, cu, 1.0 / math.sqrt(HEAD_DIM))
    _check(out, expected, BF16_ATOL, BF16_RTOL)
    # Values may change under a fixed plan; a captured graph replays correctly.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        runner()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        runner()
    q.copy_(torch.randn_like(q))
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    _check(
        out, _reference(q, k, v, cu, 1.0 / math.sqrt(HEAD_DIM)), BF16_ATOL, BF16_RTOL
    )


def test_bf16_zero_tokens():
    _require_program("bf16")
    q, k, v, cu_seqlens = _make_inputs([0, 0], 7, seed=2)
    out = minimax_h3_varlen_attention(q, k, v, cu_seqlens)
    assert tuple(out.shape) == (0, 7, HEAD_DIM)


def test_bf16_plan_cache_reuses_tables_per_layout_and_stream(monkeypatch):
    """One-shot calls with one segment layout reuse the cached plan (tables, workspace); a different
    layout, head count or stream gets its own plan; the cache is bounded and evicts the oldest."""
    _require_program("bf16")
    monkeypatch.setattr(cake_backend, "BF16_PLAN_CACHE_CAPACITY", 2)
    cake_backend._BF16_PLANS.clear()
    cu, heads = [0, 133, 300, 900], 7
    q, k, v, cu_seqlens = _make_inputs(cu, heads, seed=11)
    out = torch.empty_like(q)
    first = prepare_minimax_h3_varlen_attention(
        q, k, v, cu_seqlens, out=out, cu_seqlens_host=cu
    )
    second = prepare_minimax_h3_varlen_attention(
        k, v, q, cu_seqlens, out=out, cu_seqlens_host=cu
    )
    assert second.plan is first.plan  # same layout, same stream: cached tables
    assert len(cake_backend._BF16_PLANS) == 1
    other_heads = prepare_minimax_h3_varlen_attention(
        q[:, :5], k[:, :5], v[:, :5], cu_seqlens, cu_seqlens_host=cu
    )
    assert other_heads.plan is not first.plan
    assert len(cake_backend._BF16_PLANS) == 2
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        on_stream = prepare_minimax_h3_varlen_attention(
            q, k, v, cu_seqlens, out=out, cu_seqlens_host=cu
        )
    assert on_stream.plan is not first.plan  # streams never share a plan
    assert len(cake_backend._BF16_PLANS) == 2  # bounded: the oldest (first) was evicted
    again = prepare_minimax_h3_varlen_attention(
        q, k, v, cu_seqlens, out=out, cu_seqlens_host=cu
    )
    assert again.plan is not first.plan and len(cake_backend._BF16_PLANS) == 2
    # The cached plan reproduces the freshly built one table for table.
    fresh = build_bf16_segment_plan(cu, q.device, heads)
    for name in ("seg_begin", "seg_len", "unit_table", "combine_table"):
        assert torch.equal(getattr(again.plan, name), getattr(fresh, name)), name
    assert again.plan.num_clusters == fresh.num_clusters
    cake_backend._BF16_PLANS.clear()
    monkeypatch.undo()


def test_bf16_one_shot_repeated_calls_do_not_allocate_or_copy():
    """After the first call of a segment layout the one-shot entry re-launches with the cached plan:
    no device allocation (caller-owned ``out``) and no host-to-device copy; the output is bitwise the
    prepared runner's."""
    _require_program("bf16")
    cake_backend._BF16_PLANS.clear()
    cu, heads = [0, 4310, 4567, 4824], 7
    q, k, v, cu_seqlens = _make_inputs(cu, heads, seed=12)
    out = torch.empty_like(q)
    minimax_h3_varlen_attention(q, k, v, cu_seqlens, out=out, cu_seqlens_host=cu)
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            minimax_h3_varlen_attention(
                q, k, v, cu_seqlens, out=out, cu_seqlens_host=cu
            )
        torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == allocated
    names = [event.name for event in prof.events()]
    copies = [n for n in names if "Memcpy" in n or "memcpy" in n]
    syncs = [
        n for n in names if "Synchronize" in n and "cudaDeviceSynchronize" not in n
    ]
    assert not copies, copies
    assert not syncs, syncs
    runner = prepare_minimax_h3_varlen_attention(
        q, k, v, cu_seqlens, cu_seqlens_host=cu
    )
    runner()
    torch.cuda.synchronize()
    assert torch.equal(runner.out, out)
    _check(
        out, _reference(q, k, v, cu, 1.0 / math.sqrt(HEAD_DIM)), BF16_ATOL, BF16_RTOL
    )


@pytest.mark.parametrize(
    "label,cu,heads,halved",
    [
        ("center_5s_p2_m19296", [0, 19296], 28, {74: 28, 76: 0}),
        ("subwave_3x512_h12", [0, 512, 1024, 1536], 12, {74: 36, 76: 36}),
    ],
)
def test_bf16_q_split_plan_output_is_bitwise_the_plain_plans(
    monkeypatch, label, cu, heads, halved
):
    """A tail-round Q-split plan (the default) changes only the slot -> rows mapping: each half runs
    the whole unit's arithmetic over the same K/V range, so the output is bitwise the plain plan's.
    center_5s_p2 (1064 units) halves the 28 units of its tail round on 74 clusters and has no tail
    round on 76 (1064 = 14 x 76: the plan is the plain plan); the sub-wave layout (three 512-row
    segments x 12 heads = 36 two-stage units of four K/V blocks, for which no KV split pays) halves
    every unit into 72 single-stage halves on any grid of >= 72 clusters.  ``halved`` holds the
    expected count per grid size; other grids check the plan invariants only."""
    _require_program("bf16")
    cake_backend._BF16_PLANS.clear()
    q, k, v, cu_seqlens = _make_inputs(cu, heads, seed=13)
    split = prepare_minimax_h3_varlen_attention(q, k, v, cu_seqlens, cu_seqlens_host=cu)
    capacity = cake_backend.bf16_grid_clusters(q.device)
    units = split.plan.num_q_split_units
    if capacity in halved:
        assert units == halved[capacity], (
            f"{label}: {units} halved units on {capacity} clusters"
        )
    assert split.plan.total_tiles == heads * split.plan.total_clusters + units
    assert split.route_metadata["q_split_units"] == units
    halves = split.plan.unit_table[5::UNIT_WORDS].tolist()
    assert halves.count(1) == halves.count(2) == units
    out_split = split()
    torch.cuda.synchronize()
    monkeypatch.setattr(cake_backend, "BF16_Q_SPLIT_PLAN", False)
    plain = prepare_minimax_h3_varlen_attention(q, k, v, cu_seqlens, cu_seqlens_host=cu)
    assert plain.plan is not split.plan  # the plan cache keys the option
    assert plain.plan.num_q_split_units == 0
    assert plain.plan.total_tiles == heads * plain.plan.total_clusters
    assert not any(plain.plan.unit_table[5::UNIT_WORDS].tolist())
    out_plain = plain()
    torch.cuda.synchronize()
    assert torch.equal(out_split, out_plain)
    _check(
        out_split,
        _reference(q, k, v, cu, 1.0 / math.sqrt(HEAD_DIM)),
        BF16_ATOL,
        BF16_RTOL,
    )
    cake_backend._BF16_PLANS.clear()
    monkeypatch.undo()


ENGINE_VIEW_ROWS = [
    ("smoke_p8_133_300", [0, 133, 300], 7),
    ("empty_segments", [0, 0, 640, 640, 1200, 1201], 7),
    ("seg3_5s_p8", [0, 4310, 4567, 4824], 7),
    ("seg4_6s_p2", [0, 12285, 16401, 20393, 24384], 28),
]


@pytest.mark.parametrize("layout", ["fused_qkv", "pack"])
@pytest.mark.parametrize("label,cu,heads", ENGINE_VIEW_ROWS)
def test_bf16_engine_views_match_fp32_reference(layout, label, cu, heads):
    """Strided Q/K/V views of the fused QKV projection and of the pre-attention pack are
    consumed in place (no THD copies) on ragged multi-segment packs."""
    _require_program("bf16")
    q, k, v, cu_seqlens = _make_engine_views(cu, heads, seed=6092, layout=layout)
    out = minimax_h3_varlen_attention(q, k, v, cu_seqlens, cu_seqlens_host=cu)
    torch.cuda.synchronize()
    assert out.is_contiguous() and tuple(out.shape) == tuple(q.shape)
    _check(
        out, _reference(q, k, v, cu, 1.0 / math.sqrt(HEAD_DIM)), BF16_ATOL, BF16_RTOL
    )
    # The reference on contiguous copies of the same values agrees (the views were read
    # where they live).
    contiguous = minimax_h3_varlen_attention(
        q.contiguous(), k.contiguous(), v.contiguous(), cu_seqlens, cu_seqlens_host=cu
    )
    torch.testing.assert_close(out, contiguous, atol=0, rtol=0)


def test_bf16_rejects_head_major_view():
    """``k.transpose(0, 1).contiguous().transpose(0, 1)`` is a head-major ``[T, H, 128]``
    view with strides ``(128, T * 128, 1)``.  The kernel's operand contract is token-major
    (token stride at least ``H * head_stride``: the engine's fused-QKV column chunks and
    pack slices); head-major views are rejected up front rather than encoded into a TMA
    descriptor with non-ascending strides."""
    _require_program("bf16")
    cu, heads = [0, 300], 7
    q, k, v, cu_seqlens = _make_inputs(cu, heads, seed=3)
    k_view = k.transpose(0, 1).contiguous().transpose(0, 1)
    assert k_view.stride() == (HEAD_DIM, cu[-1] * HEAD_DIM, 1)
    with pytest.raises(ValueError, match="token stride"):
        minimax_h3_varlen_attention(q, k_view, v, cu_seqlens)


def test_bf16_rejects_bad_inputs():
    _require_program("bf16")
    q, k, v, cu_seqlens = _make_inputs([0, 300], 7, seed=3)
    with pytest.raises(ValueError, match="bfloat16"):
        minimax_h3_varlen_attention(q.float(), k, v, cu_seqlens)
    # A non-unit last stride (every other element of a [T, H, 256] buffer) is rejected.
    wide = torch.randn((300, 7, 2 * HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    k_strided = wide[:, :, ::2]
    assert tuple(k_strided.shape) == (300, 7, HEAD_DIM) and k_strided.stride(2) == 2
    with pytest.raises(ValueError, match=r"stride|contiguous"):
        minimax_h3_varlen_attention(q, k_strided, v, cu_seqlens)
    with pytest.raises(ValueError, match="token extent"):
        minimax_h3_varlen_attention(
            q, k, v, torch.tensor([0, 299], dtype=torch.int32, device="cuda")
        )
    with pytest.raises(ValueError, match="int32"):
        minimax_h3_varlen_attention(q, k, v, cu_seqlens.long())


@pytest.mark.parametrize("pv_mode", ["fp8", "fp4"])
@pytest.mark.parametrize("label,cu,heads", SMOKE_ROWS + SEGMENT_ROWS + CENTER_ROWS)
def test_nvfp4_matches_fp32_reference(pv_mode, label, cu, heads):
    _require_program(cake_backend.NVFP4_VARIANT[pv_mode])
    q, k, v, cu_seqlens = _make_inputs(cu, heads, seed=6092)
    scale = 1.0 / math.sqrt(HEAD_DIM)
    out = minimax_h3_varlen_nvfp4_attention(q, k, v, cu_seqlens, pv_mode=pv_mode)
    torch.cuda.synchronize()
    expected = _reference(q, k, v, cu, scale)
    _check(out, expected, NVFP4_ATOL, NVFP4_RTOL)
    again = minimax_h3_varlen_nvfp4_attention(q, k, v, cu_seqlens, pv_mode=pv_mode)
    torch.testing.assert_close(again, out, atol=0, rtol=0)


@pytest.mark.parametrize("pv_mode", ["fp8", "fp4"])
@pytest.mark.parametrize("label,cu,heads", LONG_ROWS)
def test_nvfp4_long_segments(pv_mode, label, cu, heads):
    _require_program(cake_backend.NVFP4_VARIANT[pv_mode])
    q, k, v, cu_seqlens = _make_inputs(cu, heads, seed=6093)
    out = minimax_h3_varlen_nvfp4_attention(
        q, k, v, cu_seqlens, pv_mode=pv_mode, cu_seqlens_host=cu
    )
    torch.cuda.synchronize()
    _check(
        out, _reference(q, k, v, cu, 1.0 / math.sqrt(HEAD_DIM)), NVFP4_ATOL, NVFP4_RTOL
    )


@pytest.mark.parametrize("pv_mode", ["fp8", "fp4"])
def test_nvfp4_prepared_runner_stages_and_graph_replay(pv_mode):
    _require_program(cake_backend.NVFP4_VARIANT[pv_mode])
    cu, heads = [0, 133, 300, 900], 7
    q, k, v, cu_seqlens = _make_inputs(cu, heads, seed=4)
    out = torch.empty_like(q)
    runner = prepare_minimax_h3_varlen_nvfp4_attention(
        q, k, v, cu_seqlens, pv_mode=pv_mode, out=out, cu_seqlens_host=cu
    )
    assert runner.plan.PB == 2 + 2 + 5 and runner.plan.total_clusters == 1 + 1 + 2
    assert runner.tile_tables.total_tiles >= heads * 4
    assert runner.route_metadata["pv_mode"] == pv_mode
    # fp8 precedes the quantizer with the ``amax`` partial-max stage.
    quantize_stages = ("amax", "quantize") if pv_mode == "fp8" else ("quantize",)
    assert tuple(runner.stage_kwargs) == quantize_stages + (
        "attention",
        "attention_split",
        "combine",
    )
    # A 3-segment plan far below one wave has no split units: the dense
    # attention program is bound, the split program and combine are not.
    assert runner.tile_tables.num_partial_slots == 0
    assert runner.attention_stage == "attention"
    assert runner.route_metadata["attention_variant"] == "attention"
    assert [name for name, entry, _ in runner._stages if entry is not None] == list(
        quantize_stages
    ) + ["attention"]
    if pv_mode == "fp8":
        assert tuple(runner.stage_kwargs["amax"]) == cake_backend.AMAX_KWARGS
        assert runner.stage_kwargs["amax"]["grid"] == (cake_backend.AMAX_CTAS, 1, 1)
        assert runner.stage_kwargs["amax"]["num_vectors"] == v.numel() // 16
    # The dense program takes the first delivery's parameter set; the split
    # program's bindings add the K/V range / partial-slot tables, the partial
    # workspace and the unit count.
    dense_names, split_names = (
        (
            cake_backend.NVFP4_ATTENTION_FP4PV_KWARGS,
            cake_backend.NVFP4_ATTENTION_SPLIT_FP4PV_KWARGS,
        )
        if pv_mode == "fp4"
        else (
            cake_backend.NVFP4_ATTENTION_FP8PV_KWARGS,
            cake_backend.NVFP4_ATTENTION_SPLIT_FP8PV_KWARGS,
        )
    )
    assert tuple(runner.stage_kwargs["attention"]) == dense_names
    assert tuple(runner.stage_kwargs["attention_split"]) == split_names
    assert set(split_names) - set(dense_names) == set(
        cake_backend.NVFP4_ATTENTION_SPLIT_KWARGS
    )
    assert runner.stage_kwargs["quantize"]["grid"] == (
        heads * 9 * QUANTIZE_SUBS_PER_BLOCK,
        1,
        1,
    )
    # Stage-wise execution equals the complete pipeline bit-exactly.
    runner.quantize()
    assert runner.attention() is out
    torch.cuda.synchronize()
    staged = out.clone()
    out.fill_(float("nan"))
    assert runner() is out
    torch.cuda.synchronize()
    torch.testing.assert_close(out, staged, atol=0, rtol=0)
    _check(
        out, _reference(q, k, v, cu, 1.0 / math.sqrt(HEAD_DIM)), NVFP4_ATOL, NVFP4_RTOL
    )
    # New values, same plan: the complete pipeline (quantize + attention) is
    # what a captured graph replays, so the output tracks the new inputs.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        runner()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        runner()
    v.copy_(torch.randn_like(v) * 3.0)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    _check(
        out, _reference(q, k, v, cu, 1.0 / math.sqrt(HEAD_DIM)), NVFP4_ATOL, NVFP4_RTOL
    )


@pytest.mark.parametrize("pv_mode", ["fp8", "fp4"])
def test_nvfp4_split_plan_binds_split_program(pv_mode):
    """A partial-wave plan binds ``attention_split`` + ``combine`` and stays exact."""
    _require_program(cake_backend.NVFP4_VARIANT[pv_mode])
    device = torch.device("cuda")
    # 12 clusters x 7 heads = 84 units: 1.14 waves on the 74-cluster grid of
    # B200/B300, which the planner splits; smaller grids split it as well.
    cu, heads = [0, 6096], 7
    q, k, v, cu_seqlens = _make_inputs(cu, heads, seed=11)
    runner = prepare_minimax_h3_varlen_nvfp4_attention(
        q, k, v, cu_seqlens, pv_mode=pv_mode, cu_seqlens_host=cu
    )
    tiles = runner.tile_tables
    if cake_backend.bf16_grid_clusters(device) >= 84:
        pytest.skip("grid holds the 84 units in one wave; no split expected")
    assert tiles.num_partial_slots > 0 and tiles.num_combine_units > 0
    assert runner.attention_stage == "attention_split"
    assert runner.route_metadata["attention_variant"] == "attention_split"
    assert [name for name, entry, _ in runner._stages if entry is not None] == (
        ["amax"] if pv_mode == "fp8" else []
    ) + ["quantize", "attention_split", "combine"]
    out = runner()
    torch.cuda.synchronize()
    _check(
        out, _reference(q, k, v, cu, 1.0 / math.sqrt(HEAD_DIM)), NVFP4_ATOL, NVFP4_RTOL
    )


def test_nvfp4_rejects_bad_pv_mode():
    _require_program("nvfp4_fp8pv")
    q, k, v, cu_seqlens = _make_inputs([0, 300], 7, seed=5)
    with pytest.raises(ValueError, match="pv_mode"):
        minimax_h3_varlen_nvfp4_attention(q, k, v, cu_seqlens, pv_mode="bf16")
