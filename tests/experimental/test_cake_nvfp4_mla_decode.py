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

Tests of the experimental NVFP4 MLA decode backend package (SM100 / SM103): the
host work planner, the decode program selector, the workspace layout, the
NVFP4 quantizer and the torch reference, the host plan of the registry records
and the ``build.json`` contract (no GPU needed), and the validated shape set on
the device.

This suite exercises the backend package alone: it names the package once, in
``PACKAGE``, and imports nothing else from flashinfer, so a build that carries
the package under another import path runs it unchanged after pointing
``PACKAGE`` at its package.  The public
``flashinfer.mla.prepare_nvfp4_batch_decode_with_kv_cache_mla`` entry is
covered by ``test_cake_nvfp4_mla_decode_api.py``.
"""

import hashlib
import importlib
import json
import math
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
import torch

PACKAGE = "flashinfer.experimental.nvfp4_mla_decode"

pkg = importlib.import_module(PACKAGE)
cake_backend = importlib.import_module(f"{PACKAGE}.cake_backend")
cake_jit = importlib.import_module(f"{PACKAGE}.cake_jit")

PACKAGE_DIR = Path(pkg.__file__).resolve().parent

BALANCED_MIN_KV = cake_backend.BALANCED_MIN_KV
BOUNDARY_COST_TILES = cake_backend.BOUNDARY_COST_TILES
CLUSTER_PAIR = cake_backend.CLUSTER_PAIR
FLAG_DIRECT_OUT = cake_backend.FLAG_DIRECT_OUT
FLAG_SEED_SINK = cake_backend.FLAG_SEED_SINK
HEAD_DIM = cake_backend.HEAD_DIM
ITEM_FIELDS = cake_backend.ITEM_FIELDS
MAX_SPLITS = cake_backend.MAX_SPLITS
PAGE_SIZE = cake_backend.PAGE_SIZE
QK_AHEAD_RECORD = cake_backend.QK_AHEAD_RECORD
REDUCE_GENERAL_RECORD = cake_backend.REDUCE_GENERAL_RECORD
REDUCE_HEADS_PER_CTA = cake_backend.REDUCE_HEADS_PER_CTA
ROW_BYTES = cake_backend.ROW_BYTES
ROWS_PER_CTA = cake_backend.ROWS_PER_CTA
ROWS_PER_TILE = cake_backend.ROWS_PER_TILE
SF_ROW_BYTES = cake_backend.SF_ROW_BYTES
SF_VEC = cake_backend.SF_VEC
SHORT_EPI_RECORD = cake_backend.SHORT_EPI_RECORD
SHORT_EPI_SPEC_RECORD = cake_backend.SHORT_EPI_SPEC_RECORD
SHORT_RECORD = cake_backend.SHORT_RECORD
SHORT_SPEC_RECORD = cake_backend.SHORT_SPEC_RECORD
V_HALVES = cake_backend.V_HALVES
XV_RECORD = cake_backend.XV_RECORD
build_work_plan = cake_backend.build_work_plan
check_pairs = cake_backend.check_pairs
host_tables = cake_backend.host_tables
kv_pages = cake_backend.kv_pages
kv_tiles = cake_backend.kv_tiles
max_nvfp4_mla_decode_workspace_size = cake_backend.max_nvfp4_mla_decode_workspace_size
nvfp4_mla_decode_workspace_size = cake_backend.nvfp4_mla_decode_workspace_size
quantize_nvfp4 = cake_backend.quantize_nvfp4
select_decode_record = cake_backend.select_decode_record
select_reduce_record = cake_backend.select_reduce_record
token_splits = cake_backend.token_splits
work_table_rows = cake_backend.work_table_rows
workspace_layout = cake_backend.workspace_layout

# The registry's further records: their stage, architectures and build
# keywords over the stage record's build.
FURTHER_RECORDS = {
    QK_AHEAD_RECORD: ("decode", ("sm_100a",), {"qk_ahead": True, "xv": True}),
    XV_RECORD: ("decode", ("sm_103a",), {"xv": True}),
    SHORT_RECORD: ("decode", cake_jit.ARCHES, {"short_rows": True}),
    SHORT_EPI_RECORD: (
        "decode",
        cake_jit.ARCHES,
        {"short_rows": True, "epi_warp": True},
    ),
    SHORT_SPEC_RECORD: (
        "decode",
        cake_jit.ARCHES,
        {"short_rows": True, "spec_uh": True},
    ),
    SHORT_EPI_SPEC_RECORD: (
        "decode",
        cake_jit.ARCHES,
        {"short_rows": True, "epi_warp": True, "spec_uh": True},
    ),
    REDUCE_GENERAL_RECORD: ("reduce", cake_jit.ARCHES, {"general_splits": True}),
}

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


def make_inputs(
    kv_lens, num_heads, *, q_len, enable_sink, device, seed=0, rng_device=None
):
    """Deterministic NVFP4 paged decode inputs with a peaked softmax.

    ``rng_device="cpu"`` draws them on the host, so every GPU sees the same
    inputs (a CUDA generator's stream depends on the device's SM count)."""
    rng = rng_device or device
    gen = torch.Generator(device=rng).manual_seed(seed)
    batch = len(kv_lens)
    pages_per_seq = [kv_pages(kv) for kv in kv_lens]
    total_pages = sum(pages_per_seq)
    max_pages = max(pages_per_seq)
    k_full = torch.randn(
        (total_pages, PAGE_SIZE, HEAD_DIM), generator=gen, device=rng
    ).to(device)
    permutation = torch.randperm(total_pages, generator=gen, device=rng).to(
        device=device, dtype=torch.int32
    )
    block_tables = torch.zeros((batch, max_pages), dtype=torch.int32, device=device)
    offset = 0
    for b, count in enumerate(pages_per_seq):
        block_tables[b, :count] = permutation[offset : offset + count]
        offset += count
    q = torch.randn((batch * q_len, num_heads, HEAD_DIM), generator=gen, device=rng).to(
        device
    )
    for b in range(batch):
        for i in range(q_len):
            visible = kv_lens[b] - q_len + i + 1
            target = int(
                torch.randint(0, visible, (1,), generator=gen, device=rng).item()
            )
            page = int(block_tables[b, target // PAGE_SIZE].item())
            q[b * q_len + i] += 0.2 * k_full[page, target % PAGE_SIZE]
    query, query_scale = quantize_nvfp4(q)
    kv_cache, kv_scale = quantize_nvfp4(k_full)
    sinks = None
    if enable_sink:
        sinks = (
            torch.randn((num_heads,), generator=gen, device=rng).to(device) * 0.5 + 1.0
        )
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


def decode_grid(plan):
    """Launch grid of the decode record for ``plan`` (the registry's grid rule)."""
    return cake_jit.launch_grid(cake_jit.MODULES["decode"], units=plan.num_units)


def assert_plan_well_formed(plan, kv_lens, *, q_len, num_heads, enable_sink):
    """Behavioural invariants every work plan must satisfy.

    Every (request, row tile, pair row) covers its pipeline tiles exactly
    once with contiguous, ordered splits; flags follow the split structure;
    unit ranges partition the table into whole row pairs of one piece; the
    launch grid is one cluster per unit.
    """
    assert (plan.q_len, plan.num_heads) == (q_len, num_heads)
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
    assert decode_grid(plan) == (CLUSTER_PAIR * plan.num_units, 1, 1)


# ---------------------------------------------------------------------------
# Host-only tests
# ---------------------------------------------------------------------------


def test_rejects_bad_shapes():
    inputs = make_inputs([64, 100], 16, q_len=6, enable_sink=False, device="cpu")
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
        decode_grid(plan)[0],
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


def _stage_of(name):
    return FURTHER_RECORDS[name][0] if name in FURTHER_RECORDS else name


def _register_further(monkeypatch, name):
    """Registers further record ``name`` (a copy of its stage record), whether
    or not the registry delivers it."""
    stage, arches, keywords = FURTHER_RECORDS[name]
    base = cake_jit.MODULES[stage]
    record = dict(
        base,
        stage=stage,
        arches=list(arches),
        build=dict(base["build"], **keywords),
    )
    monkeypatch.setitem(cake_jit.MODULES, name, record)
    return record


@pytest.fixture
def qk_ahead_record(monkeypatch):
    """Registers the SM100 QK-ahead decode record."""
    return _register_further(monkeypatch, QK_AHEAD_RECORD)


@pytest.fixture
def xv_record(monkeypatch):
    """Registers the SM103 vote-gated exchange decode record."""
    return _register_further(monkeypatch, XV_RECORD)


def _without_spec_records(monkeypatch):
    """Drops the short-row records that load their unit's first item in the
    kernel setup, so a test of the short-row rules sees the programs without."""
    for name in (SHORT_SPEC_RECORD, SHORT_EPI_SPEC_RECORD):
        monkeypatch.delitem(cake_jit.MODULES, name, raising=False)


@pytest.fixture
def short_record(monkeypatch):
    """Registers the short-row decode record for both architectures (without
    the records of ``spec_records``)."""
    _without_spec_records(monkeypatch)
    return _register_further(monkeypatch, SHORT_RECORD)


@pytest.fixture
def short_epi_record(monkeypatch):
    """Registers the per-warp epilogue short-row record for both architectures
    (without the records of ``spec_records``)."""
    _without_spec_records(monkeypatch)
    return _register_further(monkeypatch, SHORT_EPI_RECORD)


@pytest.fixture
def spec_records(monkeypatch, short_record, short_epi_record):
    """Registers both short-row records and, beside them, the two that load
    their unit's first item in the kernel setup."""
    return {
        name: _register_further(monkeypatch, name)
        for name in (SHORT_SPEC_RECORD, SHORT_EPI_SPEC_RECORD)
    }


@pytest.fixture
def general_reduce_record(monkeypatch):
    """Registers the general combine record for both architectures."""
    return _register_further(monkeypatch, REDUCE_GENERAL_RECORD)


def _bs32_plan(kv_len, num_sms, **kwargs):
    return build_work_plan(
        [kv_len] * 32,
        num_heads=64,
        num_sms=num_sms,
        q_len=6,
        enable_sink=True,
        **kwargs,
    )


def unit_tiles(plan):
    """Pipeline tiles of every unit of ``plan``."""
    return [
        sum(it[4] - it[3] for it in plan.items[a:b] if it[2] == 0)
        for a, b in zip(plan.unit_first, plan.unit_first[1:], strict=False)
    ]


def qk_ahead_geometry(plan):
    """``(min unit tiles, min longest piece)`` over the units of ``plan``."""
    units = []
    for a, b in zip(plan.unit_first, plan.unit_first[1:], strict=False):
        lens = [it[4] - it[3] for it in plan.items[a:b] if it[2] == 0]
        units.append((sum(lens), max(lens)))
    return min(u[0] for u in units), min(u[1] for u in units)


@pytest.mark.parametrize(
    "kv_len,geometry,selected",
    [
        (8192, None, False),
        (16384, None, False),
        (32768, (330, 168), False),
        (51200, (501, 264), False),
        (52224, (520, 268), True),
        (65536, (649, 333), True),
        (131072, (1327, 665), True),
    ],
)
def test_select_decode_record_picks_qk_ahead_for_long_balanced_plans(
    qk_ahead_record, kv_len, geometry, selected
):
    plan = _bs32_plan(kv_len, 148)
    assert plan.schedule == "balanced"
    if geometry is not None:
        assert qk_ahead_geometry(plan) == geometry
    want = QK_AHEAD_RECORD if selected else "decode"
    assert select_decode_record(plan, "sm_100a") == want


@pytest.mark.parametrize(
    "arch,kv_len,num_sms,schedule,registered",
    [
        ("sm_103a", 65536, 148, "auto", True),
        ("sm_103a", 131072, 148, "auto", True),
        ("sm_103a", 65536, 152, "auto", True),
        ("sm_103a", 131072, 152, "auto", True),
        ("sm_100a", 65536, 148, "auto", False),
        ("sm_100a", 131072, 148, "auto", False),
        ("sm_100a", 131072, 148, "uniform", True),
    ],
)
def test_select_decode_record_keeps_decode(
    monkeypatch, qk_ahead_record, arch, kv_len, num_sms, schedule, registered
):
    # Without the SM103 xv record (its rule: test_select_decode_record_xv_*).
    monkeypatch.delitem(cake_jit.MODULES, XV_RECORD, raising=False)
    if not registered:
        monkeypatch.delitem(cake_jit.MODULES, QK_AHEAD_RECORD, raising=False)
    plan = _bs32_plan(kv_len, num_sms, schedule=schedule, tiles_per_split=None)
    assert plan.schedule == ("uniform" if schedule == "uniform" else "balanced")
    assert select_decode_record(plan, arch) == "decode"


def _hand_plan(*units, q_len=6, num_heads=64):
    """Balanced plan whose unit ``u`` holds one piece per length in ``units[u]``."""
    items, unit_first = [], [0]
    for lens in units:
        for b, tiles in enumerate(lens):
            for pair_row in range(V_HALVES):
                items.append((b, 0, pair_row, 0, tiles, 0, 1, FLAG_DIRECT_OUT))
        unit_first.append(len(items))
    plan = cake_backend.WorkPlan(
        schedule="balanced",
        items=tuple(items),
        unit_first=tuple(unit_first),
        max_splits=1,
        tiles_per_split=None,
        q_len=q_len,
        num_heads=num_heads,
    )
    check_pairs(plan)
    return plan


@pytest.mark.parametrize("piece_tiles", [256, 257, 258])
@pytest.mark.parametrize("unit_tiles", [511, 512, 513])
def test_select_decode_record_thresholds(qk_ahead_record, unit_tiles, piece_tiles):
    assert cake_backend.QK_AHEAD_MIN_UNIT_TILES == 512
    assert cake_backend.QK_AHEAD_MIN_PIECE_TILES == 257
    rest = unit_tiles - piece_tiles
    plan = _hand_plan([piece_tiles, rest // 2, rest - rest // 2])
    want = QK_AHEAD_RECORD if unit_tiles >= 512 and piece_tiles >= 257 else "decode"
    assert select_decode_record(plan, "sm_100a") == want


def test_select_decode_record_needs_every_unit(qk_ahead_record):
    long = [258, 127, 128]
    assert select_decode_record(_hand_plan(long, long), "sm_100a") == QK_AHEAD_RECORD
    assert select_decode_record(_hand_plan(long, [256, 256]), "sm_100a") == "decode"
    assert select_decode_record(_hand_plan(long, [511]), "sm_100a") == "decode"


SHORT_LIMITS = [("sm_100a", 32), ("sm_103a", 64)]


@pytest.mark.parametrize("arch,limit", SHORT_LIMITS)
@pytest.mark.parametrize("extra", [-1, 0, 1])
def test_select_decode_record_short_threshold(short_record, arch, limit, extra):
    assert dict(SHORT_LIMITS) == cake_backend.SHORT_MAX_UNIT_TILES
    tiles = limit + extra
    plan = _hand_plan([tiles // 2, tiles - tiles // 2], [1])
    want = SHORT_RECORD if tiles <= limit else "decode"
    assert select_decode_record(plan, arch) == want


def test_select_decode_record_short_needs_every_unit(short_record):
    plan = _hand_plan([64], [32, 32], [1])
    assert select_decode_record(plan, "sm_103a") == SHORT_RECORD
    assert select_decode_record(_hand_plan([64], [32, 33]), "sm_103a") == "decode"


@pytest.mark.parametrize("arch,limit", SHORT_LIMITS)
@pytest.mark.parametrize("extra", [0, 1])
def test_select_decode_record_short_runs_uniform_plans(
    short_record, arch, limit, extra
):
    plan = build_work_plan(
        [(limit + 1) * cake_backend.TILE_KV] * 4,
        num_heads=64,
        num_sms=148,
        q_len=6,
        enable_sink=True,
        schedule="uniform",
        tiles_per_split=limit + extra,
    )
    assert max(unit_tiles(plan)) == limit + extra
    want = SHORT_RECORD if extra == 0 else "decode"
    assert select_decode_record(plan, arch) == want


@pytest.mark.parametrize("num_sms", [148, 152])
@pytest.mark.parametrize("kv_len,heaviest", [(4096, 32), (4160, 33)])
def test_select_decode_record_short_boundary_of_the_auto_uniform_schedule(
    short_record, num_sms, kv_len, heaviest
):
    """32 requests below the balanced threshold: the auto schedule runs one
    split per request, so its heaviest unit holds the request's tiles."""
    plan = _bs32_plan(kv_len, num_sms)
    assert plan.schedule == "uniform" and plan.max_splits == 1
    assert plan.tiles_per_split == kv_tiles(kv_len) == heaviest
    assert max(unit_tiles(plan)) == heaviest
    for arch, limit in SHORT_LIMITS:
        want = SHORT_RECORD if heaviest <= limit else "decode"
        assert select_decode_record(plan, arch) == want


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a"])
@pytest.mark.parametrize(
    "num_sms,batch,kv_len,heaviest",
    [
        (148, 23, 8448, 64),
        (148, 23, 8512, 65),
        (152, 24, 8320, 64),
        (152, 24, 8384, 65),
    ],
)
def test_select_decode_record_short_boundary_of_balanced_plans(
    short_record, arch, num_sms, batch, kv_len, heaviest
):
    plan = build_work_plan(
        [kv_len] * batch, num_heads=64, num_sms=num_sms, q_len=6, enable_sink=True
    )
    assert plan.schedule == "balanced"
    assert max(unit_tiles(plan)) == heaviest
    limit = cake_backend.SHORT_MAX_UNIT_TILES[arch]
    want = SHORT_RECORD if heaviest <= limit else "decode"
    assert select_decode_record(plan, arch) == want


@pytest.mark.parametrize(
    "arch,xv,want",
    [
        ("sm_100a", True, QK_AHEAD_RECORD),
        ("sm_103a", True, XV_RECORD),
        ("sm_103a", False, SHORT_RECORD),
    ],
)
def test_select_decode_record_tries_qk_ahead_and_xv_before_short(
    monkeypatch, qk_ahead_record, xv_record, short_record, arch, xv, want
):
    if not xv:
        monkeypatch.delitem(cake_jit.MODULES, XV_RECORD)
    limits = dict.fromkeys(cake_jit.ARCHES, 1 << 30)
    monkeypatch.setattr(cake_backend, "SHORT_MAX_UNIT_TILES", limits)
    assert select_decode_record(_bs32_plan(131072, 148), arch) == want


@pytest.mark.parametrize("schedule", ["auto", "uniform"])
def test_select_decode_record_tries_qk_ahead_before_xv(
    qk_ahead_record, xv_record, schedule
):
    """Where both are delivered, a balanced plan of the QK-ahead rule runs
    the QK-ahead record and a uniform one the xv record."""
    xv_record["arches"] = list(cake_jit.ARCHES)
    plan = _bs32_plan(131072, 148, schedule=schedule, tiles_per_split=None)
    want = QK_AHEAD_RECORD if plan.schedule == "balanced" else XV_RECORD
    assert select_decode_record(plan, "sm_100a") == want
    assert select_decode_record(plan, "sm_103a") == XV_RECORD


def xv_steady_tiles(plan):
    """Fewest pipeline tiles of a unit of ``plan`` past the warm-up of each
    of its pieces (the xv rule's measure)."""
    return min(
        sum(
            max(0, it[4] - it[3] - cake_backend.XV_WARMUP_TILES)
            for it in plan.items[a:b:V_HALVES]
        )
        for a, b in zip(plan.unit_first, plan.unit_first[1:], strict=False)
    )


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a"])
@pytest.mark.parametrize(
    "pieces,steady",
    [
        ([354], 306),
        ([355], 307),
        ([356], 308),
        ([200, 202, 48], 306),
        ([200, 203, 48], 307),
        ([200, 203, 47, 10], 307),
        ([47] * 12, 0),
    ],
)
def test_select_decode_record_xv_threshold(xv_record, arch, pieces, steady):
    assert cake_backend.XV_WARMUP_TILES == 48
    assert cake_backend.XV_MIN_STEADY_TILES == 307
    plan = _hand_plan(pieces, [400])
    assert xv_steady_tiles(plan) == steady
    want = XV_RECORD if arch == "sm_103a" and steady >= 307 else "decode"
    assert select_decode_record(plan, arch) == want
    uniform = replace(plan, schedule="uniform")
    assert select_decode_record(uniform, arch) == want


def test_select_decode_record_xv_needs_every_unit_and_the_record(
    monkeypatch, xv_record
):
    long = [355]
    assert select_decode_record(_hand_plan(long, long), "sm_103a") == XV_RECORD
    assert select_decode_record(_hand_plan(long, [354]), "sm_103a") == "decode"
    assert select_decode_record(_hand_plan(long, [600, 47]), "sm_103a") == XV_RECORD
    assert select_decode_record(_hand_plan(long, [47] * 20), "sm_103a") == "decode"
    xv_record["arches"] = ["sm_100a"]
    assert select_decode_record(_hand_plan(long), "sm_103a") == "decode"
    monkeypatch.delitem(cake_jit.MODULES, XV_RECORD)
    assert select_decode_record(_hand_plan(long), "sm_103a") == "decode"


@pytest.mark.parametrize(
    "num_sms,batch,kv_len,steady",
    [
        (148, 32, 32768, 208),
        (148, 32, 49152, 352),
        (148, 32, 65536, 519),
        (148, 32, 131072, 1183),
        (148, 16, 79488, 306),
        (148, 16, 79616, 307),
        (148, 16, 81920, 318),
        (148, 64, 24576, 305),
        (152, 32, 32768, 208),
        (152, 32, 49152, 339),
        (152, 32, 65536, 501),
        (152, 32, 131072, 1148),
        (152, 16, 81664, 306),
        (152, 16, 81792, 307),
        (152, 16, 81920, 307),
        (152, 64, 16384, 160),
        (152, 64, 24576, 292),
    ],
)
def test_select_decode_record_xv_boundary_of_balanced_plans(
    xv_record, num_sms, batch, kv_len, steady
):
    plan = build_work_plan(
        [kv_len] * batch, num_heads=64, num_sms=num_sms, q_len=6, enable_sink=True
    )
    assert plan.schedule == "balanced"
    assert xv_steady_tiles(plan) == steady
    want = XV_RECORD if steady >= 307 else "decode"
    assert select_decode_record(plan, "sm_103a") == want
    assert select_decode_record(plan, "sm_100a") != XV_RECORD


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a"])
@pytest.mark.parametrize(
    "num_heads,q_len,tail_rows",
    [
        (64, 6, 0),
        (128, 1, 0),
        (8, 3, 24),
        (1, 31, 31),
        (1, 32, 32),
        (8, 4, 32),
        (8, 6, 48),
        (16, 6, 32),
        (2, 31, 62),
        (64, 1, 0),
    ],
)
def test_select_decode_record_short_epi_threshold(
    short_record, short_epi_record, arch, num_heads, q_len, tail_rows
):
    assert cake_backend.SHORT_EPI_MIN_TAIL_ROWS == 32
    assert (q_len * num_heads) % ROWS_PER_CTA == tail_rows
    plan = _hand_plan([8], [8], q_len=q_len, num_heads=num_heads)
    want = SHORT_EPI_RECORD if tail_rows >= 32 else SHORT_RECORD
    assert select_decode_record(plan, arch) == want
    long = _hand_plan([400], q_len=q_len, num_heads=num_heads)
    assert select_decode_record(long, arch) not in (SHORT_RECORD, SHORT_EPI_RECORD)


def test_select_decode_record_short_epi_needs_both_records(
    monkeypatch, short_record, short_epi_record
):
    plan = _hand_plan([8], [8], q_len=4, num_heads=8)
    assert select_decode_record(plan, "sm_100a") == SHORT_EPI_RECORD
    short_epi_record["arches"] = ["sm_103a"]
    assert select_decode_record(plan, "sm_100a") == SHORT_RECORD
    assert select_decode_record(plan, "sm_103a") == SHORT_EPI_RECORD
    monkeypatch.delitem(cake_jit.MODULES, SHORT_RECORD)
    assert select_decode_record(plan, "sm_103a") == "decode"
    monkeypatch.delitem(cake_jit.MODULES, SHORT_EPI_RECORD)
    assert select_decode_record(plan, "sm_103a") == "decode"


EPI_BOUNDARY_KV = [1000, 8192, 5000, 300, 2048, 777]
SPLIT16 = {"schedule": "uniform", "tiles_per_split": 16}


@pytest.mark.parametrize("num_sms", [148, 152])
@pytest.mark.parametrize(
    "num_heads,q_len,tail_rows", [(1, 31, 31), (1, 32, 32), (8, 3, 24), (8, 4, 32)]
)
def test_select_decode_record_short_epi_boundary_of_uniform_plans(
    short_record, short_epi_record, num_sms, num_heads, q_len, tail_rows
):
    """Uniform plans of 16 pipeline tiles per split hold short units on both
    architectures; the tail rows of a request pick the short-row program."""
    plan = build_work_plan(
        EPI_BOUNDARY_KV,
        num_heads=num_heads,
        num_sms=num_sms,
        q_len=q_len,
        enable_sink=True,
        schedule="uniform",
        tiles_per_split=16,
    )
    assert max(unit_tiles(plan)) <= 16
    assert (q_len * num_heads) % ROWS_PER_CTA == tail_rows
    want = SHORT_EPI_RECORD if tail_rows >= 32 else SHORT_RECORD
    for arch in cake_jit.ARCHES:
        assert select_decode_record(plan, arch) == want


@pytest.mark.parametrize(
    "num_heads,q_len,epi", [(64, 6, False), (64, 1, False), (8, 4, True), (8, 6, True)]
)
def test_select_decode_record_spec_uh_needs_one_item_pair_per_unit(
    spec_records, num_heads, q_len, epi
):
    """The programs loading their unit's first item in the kernel setup run a
    short plan whose every unit holds one item pair; a unit of several pairs
    (or of none) keeps the program without that load."""
    short, spec = (
        (SHORT_EPI_RECORD, SHORT_EPI_SPEC_RECORD)
        if epi
        else (SHORT_RECORD, SHORT_SPEC_RECORD)
    )
    one = _hand_plan([8], [16], [1], q_len=q_len, num_heads=num_heads)
    several = _hand_plan([8], [4, 4], q_len=q_len, num_heads=num_heads)
    empty = _hand_plan([8], [], q_len=q_len, num_heads=num_heads)
    assert select_decode_record(one, "sm_100a") == spec
    assert select_decode_record(several, "sm_100a") == short
    assert select_decode_record(empty, "sm_100a") == short
    assert select_decode_record(one, "sm_103a") == spec
    assert select_decode_record(several, "sm_103a") == short
    # The short-row rule first: a long unit runs neither.
    long = _hand_plan([8], [400], q_len=q_len, num_heads=num_heads)
    assert select_decode_record(long, "sm_100a") not in (short, spec)


def test_select_decode_record_spec_uh_needs_the_records(monkeypatch, spec_records):
    one = _hand_plan([8], [8])
    one_epi = _hand_plan([8], [8], q_len=4, num_heads=8)
    assert select_decode_record(one, "sm_100a") == SHORT_SPEC_RECORD
    assert select_decode_record(one_epi, "sm_100a") == SHORT_EPI_SPEC_RECORD
    spec_records[SHORT_EPI_SPEC_RECORD]["arches"] = ["sm_103a"]
    assert select_decode_record(one_epi, "sm_100a") == SHORT_EPI_RECORD
    monkeypatch.delitem(cake_jit.MODULES, SHORT_SPEC_RECORD)
    assert select_decode_record(one, "sm_100a") == SHORT_RECORD
    # Only in place of the short-row records: without them the plan runs the decode.
    spec_records[SHORT_EPI_SPEC_RECORD]["arches"] = list(cake_jit.ARCHES)
    monkeypatch.delitem(cake_jit.MODULES, SHORT_RECORD)
    assert select_decode_record(one_epi, "sm_103a") == "decode"


SPEC_PAIRS_KV = [300, 600, 900, 1200, 9000]
# Inputs of the item-pair rows, drawn on the host so every GPU sees the same
# values (seed 15: the largest margin to the reference bound of the seeds tried).
SPEC_PAIRS_INPUTS = {"seed": 15, "rng_device": "cpu"}


@pytest.mark.parametrize("num_sms", [148, 152])
@pytest.mark.parametrize("num_heads,q_len,epi", [(64, 6, False), (8, 4, True)])
def test_select_decode_record_spec_uh_boundary_of_balanced_plans(
    spec_records, num_sms, num_heads, q_len, epi
):
    """Four short requests beside a long one: the balanced partition joins the
    end of one short request and the start of the next in one unit, so the
    plan keeps the programs without the setup load; the uniform plan of the
    same requests (one item pair per unit) runs them."""
    short, spec = (
        (SHORT_EPI_RECORD, SHORT_EPI_SPEC_RECORD)
        if epi
        else (SHORT_RECORD, SHORT_SPEC_RECORD)
    )
    kwargs = dict(num_heads=num_heads, num_sms=num_sms, q_len=q_len, enable_sink=True)
    balanced = build_work_plan(SPEC_PAIRS_KV, **kwargs)
    assert balanced.schedule == "balanced"
    pairs = [
        (last - first) // V_HALVES
        for first, last in zip(
            balanced.unit_first, balanced.unit_first[1:], strict=False
        )
    ]
    assert min(pairs) == 1 and max(pairs) > 1
    uniform = build_work_plan(
        SPEC_PAIRS_KV, schedule="uniform", tiles_per_split=16, **kwargs
    )
    for arch in cake_jit.ARCHES:
        limit = cake_backend.SHORT_MAX_UNIT_TILES[arch]
        assert max(unit_tiles(balanced)) <= limit and max(unit_tiles(uniform)) <= limit
        assert select_decode_record(balanced, arch) == short
        want = spec if arch in FURTHER_RECORDS[spec][1] else short
        assert select_decode_record(uniform, arch) == want


def test_select_decode_record_short_needs_the_record_and_a_threshold(
    monkeypatch, short_record
):
    plan = _hand_plan([8], [8])
    assert select_decode_record(plan, "sm_100a") == SHORT_RECORD
    monkeypatch.setattr(cake_backend, "SHORT_MAX_UNIT_TILES", {"sm_100a": 64})
    assert select_decode_record(plan, "sm_103a") == "decode"
    short_record["arches"] = ["sm_103a"]
    assert select_decode_record(plan, "sm_100a") == "decode"
    monkeypatch.delitem(cake_jit.MODULES, SHORT_RECORD)
    assert select_decode_record(plan, "sm_100a") == "decode"


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a"])
@pytest.mark.parametrize("max_splits", [1, 2, 3, 4, MAX_SPLITS])
def test_select_reduce_record_threshold(general_reduce_record, arch, max_splits):
    assert cake_backend.REDUCE_GENERAL_MIN_SPLITS == 3
    plan = replace(_hand_plan([8]), max_splits=max_splits)
    want = REDUCE_GENERAL_RECORD if max_splits >= 3 else "reduce"
    assert select_reduce_record(plan, arch) == want


@pytest.mark.parametrize("tiles_per_split,max_splits", [(2, 2), (1, 3)])
def test_select_reduce_record_follows_the_most_split_row(
    general_reduce_record, tiles_per_split, max_splits
):
    plan = build_work_plan(
        [256, 300],
        num_heads=64,
        num_sms=148,
        q_len=6,
        enable_sink=True,
        tiles_per_split=tiles_per_split,
    )
    assert plan.schedule == "uniform" and plan.max_splits == max_splits
    want = REDUCE_GENERAL_RECORD if max_splits >= 3 else "reduce"
    for arch in cake_jit.ARCHES:
        assert select_reduce_record(plan, arch) == want


def test_select_reduce_record_needs_the_record(monkeypatch, general_reduce_record):
    plan = replace(_hand_plan([8]), max_splits=4)
    assert select_reduce_record(plan, "sm_103a") == REDUCE_GENERAL_RECORD
    general_reduce_record["arches"] = ["sm_100a"]
    assert select_reduce_record(plan, "sm_103a") == "reduce"
    monkeypatch.delitem(cake_jit.MODULES, REDUCE_GENERAL_RECORD)
    assert select_reduce_record(plan, "sm_100a") == "reduce"


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
# Registry records and the build manifest (no GPU needed)
# ---------------------------------------------------------------------------

# Keyword names the backend binds each stage with (``grid`` excluded).
STAGE_KWARGS = {
    "decode": set(cake_backend.MAIN_KWARGS) - {"grid"},
    "reduce": set(cake_backend.REDUCE_KWARGS) - {"grid"},
}
GRID_PLAN = [("grid", "grid_x"), ("grid", "grid_y"), ("grid", "grid_z")]


def test_architectures_of_the_registry():
    assert cake_jit.ARCHES == ("sm_100a", "sm_103a")
    assert cake_jit.ARCH_BY_CAPABILITY == {(10, 0): "sm_100a", (10, 3): "sm_103a"}
    assert cake_backend.SUPPORTED_COMPUTE_CAPABILITIES == cake_jit.ARCH_BY_CAPABILITY
    assert cake_jit.STAGES == ("decode", "reduce")
    # Every stage has the record named after it; the other records are the
    # stages' optional further programs.
    assert set(cake_jit.STAGES) <= set(cake_jit.MODULES)
    assert set(cake_jit.MODULES) <= {*cake_jit.STAGES, *FURTHER_RECORDS}
    for name, record in cake_jit.MODULES.items():
        assert record["stage"] == _stage_of(name), name
    assert cake_jit.arch_for_capability((10, 0)) == "sm_100a"
    assert cake_jit.arch_for_capability((10, 3)) == "sm_103a"
    for capability in [(9, 0), (10, 1), (10, 7), (11, 0), (12, 0), (10,)]:
        assert cake_jit.arch_for_capability(capability) is None, capability
    for arch in cake_jit.ARCHES:
        assert cake_jit.program_available(arch)
    assert not cake_jit.program_available("sm_120a")
    with pytest.raises(RuntimeError, match="not delivered for sm_120a"):
        cake_jit.library_name("decode", "sm_120a")


@pytest.mark.parametrize("name", list(cake_jit.MODULES))
def test_records_carry_the_host_plan(name):
    record = cake_backend._check_record(name)
    stage = record["stage"]
    assert record is cake_jit.MODULES[name]
    assert stage == _stage_of(name)
    assert record["library"] == record["kernel"]
    assert re.fullmatch(r"cake_nvfp4_mla_decode_[0-9a-f]{20}", record["library"])
    assert record["kernel_symbol"] == "kernel_" + record["library"]
    assert record["ffi_entry"] == "run"
    arches = FURTHER_RECORDS[name][1] if name in FURTHER_RECORDS else cake_jit.ARCHES
    assert list(record["arches"]) == list(arches)
    assert not {"sources", "source_sha256", "compile_flags", "cache_names"} & set(
        record
    )
    launch = record["launch"]
    assert launch["use_pdl"] is False and launch["cooperative"] is False
    assert tuple(launch["cluster"]) == ((2, 1, 1) if stage == "decode" else (1, 1, 1))
    plan = [tuple(item) for item in record["arg_plan"]]
    assert plan[-3:] == GRID_PLAN
    assert all(kind in ("tma_buffer", "buffer", "parameter") for kind, _ in plan[:-3])
    assert set(record["logical"]) == {key for _, key in plan[:-3]}
    assert set(record["logical"].values()) <= STAGE_KWARGS[stage]


def test_check_record_refuses_a_different_host_plan(monkeypatch):
    record = dict(cake_jit.MODULES["decode"], tile_kv=2 * cake_backend.TILE_KV)
    monkeypatch.setitem(cake_jit.MODULES, "decode", record)
    with pytest.raises(RuntimeError, match="host plan"):
        cake_backend._check_record("decode")


@pytest.mark.parametrize("name", list(FURTHER_RECORDS))
def test_check_record_holds_a_record_to_its_stage(monkeypatch, name):
    record = _register_further(monkeypatch, name)
    assert cake_backend._check_record(name) is record
    record["grid_rule"] = {"kind": "units_by_cluster", "ctas": 1}
    with pytest.raises(RuntimeError, match=f"{name} record's host plan"):
        cake_backend._check_record(name)


@pytest.mark.parametrize("name", list(FURTHER_RECORDS))
def test_further_records_share_their_stage_host_plan(name):
    record = cake_jit.MODULES.get(name)
    if record is None:
        pytest.skip(f"the registry does not deliver the {name} program")
    stage, _, keywords = FURTHER_RECORDS[name]
    base = cake_jit.MODULES[stage]
    assert record["build"] == dict(base["build"], **keywords)
    assert record["library"] != base["library"]
    shared = {"module", "stage", "ffi_entry", "arg_plan", "logical"}
    for key in shared | set(cake_backend._RECORD_CONSTANTS[stage]):
        assert record[key] == base[key], key
    assert record["launch"]["cluster"] == base["launch"]["cluster"]


@pytest.mark.parametrize("stage", ["decode", "reduce"])
def test_launch_args_follow_the_argument_plan(stage):
    record = cake_jit.MODULES[stage]
    logical = record["logical"]
    values = {
        logical[key]: (index if kind == "parameter" else f"<{key}>")
        for index, (kind, key) in enumerate(record["arg_plan"])
        if kind != "grid"
    }
    grid = (7, 11, 13)
    args = cake_jit.launch_args(record, values, grid)
    expected = tuple(
        values[logical[key]] for kind, key in record["arg_plan"] if kind != "grid"
    )
    assert args == expected + grid
    with pytest.raises(ValueError, match="needs logical argument 'num_heads'"):
        cake_jit.launch_args(
            record, {k: v for k, v in values.items() if k != "num_heads"}, grid
        )
    with pytest.raises(ValueError, match="has no logical name"):
        cake_jit.launch_args(dict(record, logical={}), values, grid)


def test_launch_grid_follows_the_grid_rules():
    decode, reduce = cake_jit.MODULES["decode"], cake_jit.MODULES["reduce"]
    ctas = decode["grid_rule"]["ctas"]
    heads_per_cta = reduce["grid_rule"]["heads_per_cta"]
    assert (ctas, heads_per_cta) == (CLUSTER_PAIR, REDUCE_HEADS_PER_CTA) == (2, 16)
    for units in [1, 9, 74]:
        assert cake_jit.launch_grid(decode, units=units) == (ctas * units, 1, 1)
    for total_q, num_heads in [(192, 64), (18, 8), (6, 16), (6, 17), (1, 128)]:
        assert cake_jit.launch_grid(reduce, total_q=total_q, num_heads=num_heads) == (
            total_q,
            math.ceil(num_heads / heads_per_cta),
            1,
        )
    with pytest.raises(ValueError, match="positive unit count"):
        cake_jit.launch_grid(decode, units=0)
    with pytest.raises(ValueError, match="positive rows and heads"):
        cake_jit.launch_grid(reduce, total_q=0, num_heads=64)
    with pytest.raises(ValueError, match="unknown Cake NVFP4 MLA decode grid rule"):
        cake_jit.launch_grid({"grid_rule": {"kind": "linear"}}, units=1)


def test_registry_records_reference_the_build_manifest():
    """Every ``MODULES`` record names its ``build.json`` libraries.

    The manifest (schema ``cake-build/1``) carries exactly one entry per (record
    library, architecture), named ``<library>_<arch>``, whose kernel and binding
    sources lie under this package with matching digests; the decode-stage entries
    record ``--use_fast_math`` and the combine entries do not.
    """
    manifest = json.loads((PACKAGE_DIR / "build.json").read_text())
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == pkg.__name__
    assert set(manifest["arch_flags"]) == set(cake_jit.ARCHES)
    entries = manifest["libraries"]
    libraries = {entry["name"]: entry for entry in entries}
    assert len(libraries) == len(entries)
    referenced = set()
    for name, record in cake_jit.MODULES.items():
        for arch in record["arches"]:
            library = cake_jit.library_name(name, arch)
            assert library == f"{record['library']}_{arch}"
            referenced.add(library)
            entry = libraries[library]
            assert entry["arch"] == arch
            assert entry["entry"] == record["ffi_entry"]
            assert [Path(source).name for source in entry["sources"]] == [
                f"{record['library']}_kernel.cu",
                f"{record['library']}_binding.cu",
            ]
            assert set(entry["sha256"]) == set(entry["sources"])
            for source in entry["sources"]:
                path = PACKAGE_DIR / source
                assert path.is_file(), source
                assert path.resolve().is_relative_to(PACKAGE_DIR), source
                assert (
                    hashlib.sha256(path.read_bytes()).hexdigest()
                    == entry["sha256"][source]
                ), source
            assert re.fullmatch(r"[0-9a-f]{64}", entry["closure_sha256"])
            fast_math = "--use_fast_math" in entry["nvcc_flags"]
            assert fast_math == (record["stage"] == "decode"), library
    assert referenced == set(libraries)


# ---------------------------------------------------------------------------
# GPU tests
# ---------------------------------------------------------------------------


def _gpu_skip_reason():
    if not torch.cuda.is_available():
        return "CUDA required"
    arch = cake_jit.arch_for_capability(torch.cuda.get_device_capability())
    if arch is None:
        return "SM100 or SM103 required"
    if not cake_jit.program_available(arch):
        return "generated programs not registered for this device"
    return None


def assert_bound_to_the_record(arguments, record, kwargs):
    """The runner's arguments are ``kwargs`` in ``arg_plan`` order, then the grid."""
    plan = record["arg_plan"]
    assert len(arguments) == len(plan)
    assert arguments[-3:] == tuple(kwargs["grid"])
    for argument, (kind, key) in zip(arguments[:-3], plan[:-3], strict=True):
        assert kind != "grid"
        assert argument is kwargs[record["logical"][key]], key


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


@pytest.mark.parametrize(
    "label,kv_lens,q_len,num_heads,enable_sink,tiles_per_split",
    [
        ("smoke_forced_two_splits", [256, 300], 6, 64, True, 2),
        ("smoke_forced_three_splits", [256, 300], 6, 64, True, 1),
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
    num_sms = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
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
    main_record = cake_jit.MODULES[select_decode_record(decode.plan, decode.arch)]
    assert decode.main_program == main_record["library"]
    assert decode.main_kwargs["grid"] == decode_grid(decode.plan)
    assert decode.main_kwargs["grid"] == (CLUSTER_PAIR * decode.plan.num_units, 1, 1)
    assert_bound_to_the_record(decode.main_arguments, main_record, decode.main_kwargs)
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
        reduce_record = cake_jit.MODULES[select_reduce_record(decode.plan, decode.arch)]
        assert decode.reduce_program == reduce_record["library"]
        assert decode.reduce_kwargs["grid"] == cake_jit.launch_grid(
            reduce_record, total_q=total_q, num_heads=num_heads
        )
        assert_bound_to_the_record(
            decode.reduce_arguments, reduce_record, decode.reduce_kwargs
        )
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


def assert_programs_match_bitwise(
    monkeypatch,
    name,
    kv_lens,
    *,
    q_len=6,
    num_heads=64,
    enable_sink=True,
    schedule="auto",
    tiles_per_split=None,
    seed=0,
    rng_device=None,
):
    """The selector binds further record ``name`` for the problem, and the
    stage's own record and ``name`` run it to the same bits."""
    reason = _gpu_skip_reason()
    if reason:
        pytest.skip(reason)
    arch = cake_jit.arch_for_capability(torch.cuda.get_device_capability())
    record = cake_jit.MODULES.get(name)
    if record is None or arch not in record["arches"]:
        pytest.skip(f"no {name} program registered for {arch}")
    stage = FURTHER_RECORDS[name][0]
    if stage == "decode":
        selector, program, bound = "select_decode_record", "main_program", "main"
    else:
        selector, program, bound = "select_reduce_record", "reduce_program", "reduce"
    num_sms = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    plan = build_work_plan(
        kv_lens,
        num_heads=num_heads,
        num_sms=num_sms,
        q_len=q_len,
        enable_sink=enable_sink,
        schedule=schedule,
        tiles_per_split=tiles_per_split,
    )
    if getattr(cake_backend, selector)(plan, arch) != name:
        pytest.skip(f"the selector keeps the {stage} program at {num_sms} SMs")
    inputs = make_inputs(
        kv_lens,
        num_heads,
        q_len=q_len,
        enable_sink=enable_sink,
        device="cuda",
        seed=seed,
        rng_device=rng_device,
    )
    if tiles_per_split is None:
        nbytes = nvfp4_mla_decode_workspace_size(
            kv_lens,
            num_heads,
            num_sms=num_sms,
            q_len=q_len,
            enable_sink=enable_sink,
            schedule=schedule,
        )
    else:
        nbytes = max_nvfp4_mla_decode_workspace_size(
            len(kv_lens), num_heads, q_len=q_len
        )
    workspace = torch.empty((nbytes,), dtype=torch.uint8, device="cuda")

    def prepare():
        workspace.fill_(0xFF)  # poisoned: see test_nvfp4_mla_decode
        rows = (len(kv_lens) * q_len, num_heads)
        return cake_backend.prepare_nvfp4_batch_decode_with_kv_cache_mla(
            inputs["query"],
            inputs["query_scale"],
            inputs["kv_cache"],
            inputs["kv_scale"],
            inputs["block_tables"],
            inputs["seq_lens"],
            workspace,
            sm_scale=inputs["sm_scale"],
            sinks=inputs["sinks"],
            out=torch.empty((*rows, HEAD_DIM), dtype=torch.bfloat16, device="cuda"),
            lse=torch.full(rows, float("nan"), dtype=torch.float32, device="cuda"),
            return_lse=True,
            schedule=schedule,
            tiles_per_split=tiles_per_split,
        )

    # The real selector binds the further program for this plan.
    selected = prepare()
    assert getattr(selected, program) == record["library"]
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
    results = []
    for key in (stage, name):
        monkeypatch.setattr(cake_backend, selector, lambda plan, arch, key=key: key)
        decode = prepare()
        bound_record = cake_jit.MODULES[key]
        assert getattr(decode, program) == bound_record["library"]
        assert_bound_to_the_record(
            getattr(decode, f"{bound}_arguments"),
            bound_record,
            getattr(decode, f"{bound}_kwargs"),
        )
        out, lse = decode()
        torch.cuda.synchronize()
        check_outputs(out, lse, ref_out, ref_lse)
        results.append((out, lse))
    (out_a, lse_a), (out_b, lse_b) = results
    assert torch.equal(out_a, out_b) and torch.equal(lse_a, lse_b)


@pytest.mark.parametrize("kv_len", [52224, 65536, 131072])
def test_qk_ahead_decode_matches_the_decode_bitwise(monkeypatch, kv_len):
    """Both SM100 decode programs run long balanced plans to the same bits."""
    assert_programs_match_bitwise(monkeypatch, QK_AHEAD_RECORD, [kv_len] * 32)


@pytest.mark.parametrize(
    "label,kv_lens",
    [
        ("bs32_q6_kv64k", [65536] * 32),
        ("bs32_q6_kv128k", [131072] * 32),
        ("bs32_q6_kv49152", [49152] * 32),
        ("bs16_q6_kv81920", [81920] * 16),
    ],
)
def test_xv_decode_matches_the_decode_bitwise(monkeypatch, label, kv_lens):
    """Both SM103 decode programs run the plans of the xv rule to the same
    bits (bs16_q6_kv81920: the rule's boundary row at 152 SMs)."""
    assert_programs_match_bitwise(monkeypatch, XV_RECORD, kv_lens)


@pytest.mark.parametrize(
    "label,kv_lens,options",
    [
        ("smoke_forced_two_splits", [256, 300], {"tiles_per_split": 2}),
        ("no_sink_q1", [64, 65, 4096, 777], {"q_len": 1, "enable_sink": False}),
        ("h8_q3_tail24", EPI_BOUNDARY_KV, {"num_heads": 8, "q_len": 3, **SPLIT16}),
        ("bs8_q6_kv8k", [8192] * 8, {}),
        ("bs32_q6_kv4096", [4096] * 32, {}),
        ("bs23_q6_kv8448", [8448] * 23, {}),
        ("bs24_q6_kv8320", [8320] * 24, {}),
        ("spec_pairs_h64_q6", SPEC_PAIRS_KV, SPEC_PAIRS_INPUTS),
    ],
)
def test_short_decode_matches_the_decode_bitwise(monkeypatch, label, kv_lens, options):
    """Plans of short units run the short-row and the decode programs to the
    same bits (the boundary rows where the device's SM count selects them)."""
    assert_programs_match_bitwise(monkeypatch, SHORT_RECORD, kv_lens, **options)


@pytest.mark.parametrize(
    "label,kv_lens,options",
    [
        ("partial_pages_h8", [1000, 8192, 5000], {"num_heads": 8}),
        ("h1_q32_tail32", EPI_BOUNDARY_KV, {"num_heads": 1, "q_len": 32, **SPLIT16}),
        ("h8_q4_tail32", EPI_BOUNDARY_KV, {"num_heads": 8, "q_len": 4, **SPLIT16}),
        ("h16_q6_tail32", [4096] * 8, {"num_heads": 16}),
        (
            "spec_pairs_h8_q4",
            SPEC_PAIRS_KV,
            {"num_heads": 8, "q_len": 4, **SPEC_PAIRS_INPUTS},
        ),
    ],
)
def test_short_epi_decode_matches_the_decode_bitwise(
    monkeypatch, label, kv_lens, options
):
    """Short plans whose requests end in a CTA of at least 32 valid query
    rows run the per-warp epilogue and the decode programs to the same bits."""
    assert_programs_match_bitwise(monkeypatch, SHORT_EPI_RECORD, kv_lens, **options)


@pytest.mark.parametrize(
    "label,kv_lens,options",
    [
        ("smoke_forced_two_splits", [256, 300], {"tiles_per_split": 2}),
        ("no_sink_q1", [64, 65, 4096, 777], {"q_len": 1, "enable_sink": False}),
        ("h1_q31_tail31", EPI_BOUNDARY_KV, {"num_heads": 1, "q_len": 31, **SPLIT16}),
        ("h8_q3_tail24", EPI_BOUNDARY_KV, {"num_heads": 8, "q_len": 3, **SPLIT16}),
        ("bs32_q6_kv4096", [4096] * 32, {}),
        ("spec_pairs_h64_q6_uniform", SPEC_PAIRS_KV, {**SPLIT16, **SPEC_PAIRS_INPUTS}),
    ],
)
def test_short_spec_decode_matches_the_decode_bitwise(
    monkeypatch, label, kv_lens, options
):
    """Short plans of one item pair per unit run the short-row program that
    loads its unit's first item in the kernel setup and the decode program to
    the same bits."""
    assert_programs_match_bitwise(monkeypatch, SHORT_SPEC_RECORD, kv_lens, **options)


@pytest.mark.parametrize(
    "label,kv_lens,options",
    [
        ("partial_pages_h8", [1000, 8192, 5000], {"num_heads": 8, **SPLIT16}),
        ("h1_q32_tail32", EPI_BOUNDARY_KV, {"num_heads": 1, "q_len": 32, **SPLIT16}),
        ("h8_q4_tail32", EPI_BOUNDARY_KV, {"num_heads": 8, "q_len": 4, **SPLIT16}),
        ("h16_q6_tail32", [4096] * 8, {"num_heads": 16}),
        (
            "spec_pairs_h8_q4_uniform",
            SPEC_PAIRS_KV,
            {"num_heads": 8, "q_len": 4, **SPLIT16, **SPEC_PAIRS_INPUTS},
        ),
    ],
)
def test_short_epi_spec_decode_matches_the_decode_bitwise(
    monkeypatch, label, kv_lens, options
):
    """Short plans of one item pair per unit whose requests end in a CTA of at
    least 32 valid query rows run the per-warp epilogue program that loads its
    unit's first item in the kernel setup and the decode program to the same
    bits."""
    assert_programs_match_bitwise(
        monkeypatch, SHORT_EPI_SPEC_RECORD, kv_lens, **options
    )


@pytest.mark.parametrize(
    "label,kv_lens,options",
    [
        ("smoke_forced_three_splits", [256, 300], {"tiles_per_split": 1}),
        ("partial_pages_h8", [1000, 8192, 5000], {"num_heads": 8}),
        ("bs4_q6_kv4k_16_splits", [4096] * 4, {"tiles_per_split": 2}),
    ],
)
def test_general_reduce_matches_the_reduce_bitwise(
    monkeypatch, label, kv_lens, options
):
    """Plans of more than two splits run both combine programs to the same bits."""
    assert_programs_match_bitwise(
        monkeypatch, REDUCE_GENERAL_RECORD, kv_lens, **options
    )
