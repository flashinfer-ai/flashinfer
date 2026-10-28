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

"""Cake SM90 VSA (``backend="cake"`` on Hopper): planning, numerics, lifetime, package contract,
the Sage FP8 route (E4M3 q/k/v with scales bound at plan time; the FP8 stages are twins of the
BF16 stages on the same planner bytes, plus the two quantizer stages) and the grouped-query route
(``num_qo_heads = g x num_kv_heads``, ``g`` in {2, 4, 8}: the GQA stages on the GQA planner's bytes,
checked against the MHA route on replicated inputs -- bit for bit on the static programs).

This suite exercises the backend package alone: it names the package once,
through ``tests.test_helpers.cake_package``, and imports nothing else from
flashinfer, so a build that vendors the package under another import path
runs it unchanged after pointing that table entry at its package.  The
``flashinfer.cake_vsa_sm90`` facade and the ``flashinfer.sparse`` wrapper
route (``backend="cake"`` / ``"cake_cute"``) are covered by
``test_cake_vsa_sm90_api.py``.
"""


import ast
import contextlib
import hashlib
import inspect
from pathlib import Path

import pytest
import torch

from tests.test_helpers.cake_package import cake_package

pkg = cake_package("cake_vsa_sm90")
cake_build = cake_package("cake_vsa_sm90", "cake_build")
loader = cake_package("cake_vsa_sm90", "cake_jit")
# The host planner: the tests below monkeypatch its thresholds (``POOL_MIN_TILES``,
# ``POOL_SMALL_TILE_POS``) on this module object.
planner = cake_package("cake_vsa_sm90", "cake_planner")

MAX_NSPLIT = planner.MAX_NSPLIT
PLAN_HALFWORDS = planner.PLAN_HALFWORDS
CLUSTER_VARIANTS = planner.CLUSTER_VARIANTS
PLAN_META_SPLIT = planner.PLAN_META_SPLIT
PLAN_META_UNSPLIT = planner.PLAN_META_UNSPLIT
SMALL_KMAX_VARIANTS = planner.SMALL_KMAX_VARIANTS
SPLIT_KMAX_VARIANTS = planner.SPLIT_KMAX_VARIANTS
SPLIT_CLUSTER_VARIANTS = planner.SPLIT_CLUSTER_VARIANTS
SPLIT_CLUSTER_MIN_SAVING_KB = planner.SPLIT_CLUSTER_MIN_SAVING_KB
SMALL_OCCUPANCY = planner.SMALL_OCCUPANCY
SMALL_ITEM_ELEMS = planner.SMALL_ITEM_ELEMS
SMALL_STATS_FLOATS = planner.SMALL_STATS_FLOATS
cluster_cost = planner.cluster_cost
cluster_capacity = planner.cluster_capacity
cluster_variant_for = planner.cluster_variant_for
split_cost = planner.split_cost
MAX_OWN = planner.MAX_OWN
META_NOWN = planner.META_NOWN
META_NSEQ = planner.META_NSEQ
META_NTILES = planner.META_NTILES
META_OWN_OFF = planner.META_OWN_OFF
META_SEQ_OFF = planner.META_SEQ_OFF
META_WORDS = planner.META_WORDS
OWN_WORDS = planner.OWN_WORDS
POOL_MIN_TILES = planner.POOL_MIN_TILES
POOL_RAGGED_LONG_HEAD_MIN_TILES = planner.POOL_RAGGED_LONG_HEAD_MIN_TILES
POOL_RAGGED_LONG_HEAD_WINDOWS = planner.POOL_RAGGED_LONG_HEAD_WINDOWS
POOL_RAGGED_MIN_TILES = planner.POOL_RAGGED_MIN_TILES
POOL_ROUNDS = planner.POOL_ROUNDS
POOL_SMALL_MIN_TILES = planner.POOL_SMALL_MIN_TILES
POOL_SMALL_TILE_POS = planner.POOL_SMALL_TILE_POS
COMPACT_MAX_TILE_POS = planner.COMPACT_MAX_TILE_POS
COMPACT_MIN_TILES = planner.COMPACT_MIN_TILES
_schedule_feasible = planner._schedule_feasible
_split_ingest_kb = planner._split_ingest_kb
plan_small = planner.plan_small
plan_vsa_sm90 = planner.plan_vsa_sm90
small_route = planner.small_route
split_cluster_variant_for = planner.split_cluster_variant_for
split_kmax = planner.split_kmax
CakeVsaSm90Plan = pkg.CakeVsaSm90Plan
create_plan = pkg.create_plan

REFERENCE_CLUSTER_CAPACITY_H100_SXM: dict[int, int] = {
    2: 66,
    3: 39,
    4: 30,
    6: 17,
    8: 15,
}


CAP = REFERENCE_CLUSTER_CAPACITY_H100_SXM


def _random_mask(h, mb, nb, capacity, seed=0, ragged=True, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    mask = torch.zeros((h, mb, nb), dtype=torch.bool)
    for head in range(h):
        for row in range(mb):
            count = (
                capacity
                if (row == 0 or not ragged)
                else 1 + (row * 7 + head) % capacity
            )
            mask[head, row, torch.randperm(nb, generator=g)[:count]] = True
    return mask.to(device)


def _descriptors(h=2, mb=3, nb=5, device="cpu"):
    mask = torch.ones((h, mb, nb), dtype=torch.bool, device=device)
    rows = torch.full((h, mb), 64, dtype=torch.int32, device=device)
    cols = torch.full((h, nb), 64, dtype=torch.int32, device=device)
    return mask, rows, cols


@pytest.mark.parametrize(
    "h,mb,nb,capacity,expected",
    [
        (1, 1, 1, 1, (1, False, 0)),  # one tile, one block
        (4, 4, 4, 1, (1, False, 0)),  # 16 tiles
        (8, 4, 4, 3, (3, False, 0)),
        (8, 16, 16, 4, (4, False, 0)),  # 128 tiles at 1 CTA/SM
        (
            8,
            17,
            16,
            4,
            None,
        ),  # 136 tiles > 132 SMs at 1 CTA/SM, too many for the split rule
        (8, 64, 64, 1, (1, False, 0)),  # 512 tiles at 4 CTAs/SM
        (8, 32, 32, 8, None),  # 8 blocks > KMAX 6, 256 tiles fill the persistent kernel
        (2, 128, 32, 6, None),  # 256 x 8 halfwords > PLAN_HALFWORDS
        (
            1,
            16,
            16,
            16,
            "cluster",
        ),  # one head, 16 x 16 blocks: 32 warpgroups < 132 SMs -> cluster (uniform selection)
        (
            2,
            8,
            64,
            32,
            "split",
        ),  # 16 tiles x 32 blocks: 6 slices of KMAX 6 = 96 items (no cluster holds 32)
        (
            2,
            8,
            64,
            64,
            "split2",
        ),  # 16 tiles x 64 blocks: no one-wave variant; the under-filled grid takes KMAX 6 split in two waves
    ],
)
def test_small_route_rule(h, mb, nb, capacity, expected):
    mask = _random_mask(h, mb, nb, capacity, ragged=False)
    route = small_route(mask, sms=132, cluster_capacity=CAP)
    counts = [capacity] * (h * mb)
    if expected == "split":
        assert route is not None and route[1] is True and route[2] == 0
        kmax = route[0]
        assert kmax in SMALL_KMAX_VARIANTS and 2 * h * mb <= 132
        items = h * mb * -(-capacity // kmax)
        assert items <= SMALL_OCCUPANCY[kmax] * 132
        assert items * (kmax + PLAN_META_SPLIT) <= PLAN_HALFWORDS
        assert kmax == split_kmax(counts, sms=132)
    elif expected == "cluster":
        assert route is not None and route[1] is False and route[2] > 0
        kmax, csize = route[0], route[2]
        assert (kmax, csize) in CLUSTER_VARIANTS and kmax * csize >= capacity
        assert h * mb <= CAP[csize]  # one wave of clusters on the 132-SM H100 SXM
        assert (kmax, csize) == cluster_variant_for(
            counts, sms=132, cluster_capacity=CAP
        )
        assert cluster_cost(counts, kmax, csize) < split_cost(
            counts, split_kmax(counts, sms=132)
        )
    elif expected == "split2":
        # A persistent grid of at most SMs / SPLIT_UNDERFILL_SMS
        # CTAs whose selection no one-wave split / cluster variant holds takes the split-KV
        # variant in up to SPLIT_UNDERFILL_WAVES waves of items (fewest items first).
        assert route is not None and route[1] is True and route[2] == 0
        kmax = route[0]
        assert (
            kmax in SPLIT_KMAX_VARIANTS and h * mb * planner.SPLIT_UNDERFILL_SMS <= 132
        )
        items = h * mb * -(-capacity // kmax)
        assert SMALL_OCCUPANCY[kmax] * 132 < items
        assert items <= planner.SPLIT_UNDERFILL_WAVES * SMALL_OCCUPANCY[kmax] * 132
        assert items * (kmax + PLAN_META_SPLIT) <= PLAN_HALFWORDS
        assert kmax == split_kmax(counts, sms=132, waves=planner.SPLIT_UNDERFILL_WAVES)
        with pytest.raises(ValueError):
            split_kmax(counts, sms=132)
    else:
        assert route == expected
    if expected not in (None, "split", "split2", "cluster"):
        assert h * mb <= SMALL_OCCUPANCY[expected[0]] * 132
        assert h * mb * (expected[0] + PLAN_META_UNSPLIT) <= PLAN_HALFWORDS


def test_sliced_routes_take_the_cheaper_modelled_variant():
    """Below the persistent kernel's occupancy the split and cluster variants are compared by
    their modelled cost: ragged 1..8 selections over 64 query blocks take the two-CTA cluster
    (two-warpgroup chain of two blocks, one DSM merge) and 16 uniform selections the 4 x 4 cluster."""
    ragged = torch.zeros((4, 16, 16), dtype=torch.bool)
    for t in range(64):
        ragged.reshape(64, 16)[t, : 1 + t % 8] = True
    counts = ragged.sum(-1).reshape(-1).tolist()
    kmax, split, cluster = small_route(ragged, sms=132, cluster_capacity=CAP)
    assert (kmax, split, cluster) == (4, False, 2)
    assert split_kmax(counts, sms=132) in (
        3,
        4,
        6,
    )  # a split variant fits; it is just costlier
    assert cluster_variant_for(counts, sms=132, cluster_capacity=CAP) == (4, 2)
    uniform = torch.zeros((1, 16, 16), dtype=torch.bool)
    uniform[..., :16] = True
    assert small_route(uniform, sms=132, cluster_capacity=CAP) == (4, False, 4)


def test_cluster_one_wave_uses_the_device_capacity():
    """The 132-SM H100 holds 30 four-CTA clusters (not 8 GPCs x 4 = 32): 31-32 tiles of capacity 7-8
    take the two-CTA (4, 2) variant instead of a second wave of (2, 4); capacity 13-16 has no
    one-wave cluster there and falls back to the split route; the capacity table is mandatory."""
    assert cluster_variant_for([8] * 30, sms=132, cluster_capacity=CAP) == (2, 4)
    assert cluster_variant_for([8] * 31, sms=132, cluster_capacity=CAP) == (4, 2)
    assert cluster_variant_for([8] * 32, sms=132, cluster_capacity=CAP) == (4, 2)
    assert cluster_variant_for([16] * 31, sms=132, cluster_capacity=CAP) is None
    mask = torch.zeros((8, 4, 64), dtype=torch.bool)
    mask[..., :16] = True  # 32 tiles x 16 blocks
    kmax, split, cluster = small_route(mask, sms=132, cluster_capacity=CAP)
    assert split is True and cluster == 0 and kmax == split_kmax([16] * 32, sms=132)
    with pytest.raises(ValueError, match="cluster_capacity_unresolved"):
        cluster_variant_for([8] * 16, sms=132)
    with pytest.raises(ValueError, match="cluster_capacity_missing"):
        cluster_variant_for([8] * 16, sms=132, cluster_capacity={2: 66})


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0),
    reason="thread block clusters need a Hopper or newer GPU",
)
def test_cluster_capacity_resolves_from_the_driver():
    cap = cluster_capacity(torch.cuda.current_device())
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    assert set(cap) == {
        2,
        3,
        4,
        6,
        8,
    }  # the DSM-cluster sizes and the split-cluster size 8
    for csize, clusters in cap.items():
        assert 1 <= clusters <= sms // csize
    assert cap[4] <= cap[2] // 2 + 1
    assert cap[8] <= cap[4] // 2 + 1


def test_plan_small_cluster_rows_pad_every_query_block():
    mask = _random_mask(2, 8, 64, 16, seed=7, ragged=True)
    for kmax, csize in CLUSTER_VARIANTS:
        if kmax * csize < 16:
            continue
        plan = plan_small(mask, kmax=kmax, cluster=csize)
        rows = _decode_small_rows(plan)
        assert plan["cluster"] == csize and not plan["split"]
        assert plan["num_items"] == len(rows) == 16 * csize
        for tile in range(16):
            selected = torch.nonzero(mask.reshape(16, 64)[tile]).flatten().tolist()
            slices = rows[tile * csize : (tile + 1) * csize]
            assert all(r["tile"] == tile and r["nsplit"] == csize for r in slices)
            assert [r["split"] for r in slices] == list(range(csize))
            assert all(0 <= r["count"] == len(r["blocks"]) <= kmax for r in slices)
            assert sorted(b for r in slices for b in r["blocks"]) == selected
    with pytest.raises(ValueError, match="cannot hold"):
        plan_small(torch.ones((1, 16, 16), dtype=torch.bool), kmax=4, cluster=2)
    # split + cluster names the split-cluster variants only
    with pytest.raises(ValueError, match="not a compiled split-cluster variant"):
        plan_small(
            torch.ones((1, 16, 16), dtype=torch.bool), kmax=4, split=True, cluster=4
        )


def _decode_small_rows(plan):
    stride, meta_halfwords = plan["stride"], plan["meta_halfwords"]
    slots = (
        plan["kmax"] * plan["cluster"]
        if plan["split"] and plan["cluster"]
        else plan["kmax"]
    )
    assert stride == slots + meta_halfwords
    rows = plan["plan"][
        : (
            plan["num_tiles"]
            if plan["split"] and plan["cluster"]
            else plan["num_items"]
        )
        * stride
    ]
    rows = rows.view(-1, stride).int()
    out = []
    for i in range(rows.shape[0]):
        meta = int(rows[i, 0])
        if meta_halfwords == PLAN_META_UNSPLIT:  # [count, blk...] per query block
            out.append(
                {
                    "count": meta,
                    "split": 0,
                    "nsplit": 1,
                    "tile": i,
                    "blocks": [int(b) for b in rows[i, 1:] if int(b) >= 0],
                }
            )
            continue
        out.append(
            {
                "count": meta & 15,
                "split": (meta >> 4) & 63,
                "nsplit": meta >> 10,
                "tile": int(rows[i, 1]),
                "blocks": [int(b) for b in rows[i, meta_halfwords:] if int(b) >= 0],
            }
        )
    return out


def test_plan_small_layout_and_padding():
    mask = _random_mask(3, 5, 9, 4, seed=3, ragged=True)
    plan = plan_small(mask)
    assert plan["kmax"] == 4 and plan["num_tiles"] == plan["num_items"] == 15
    assert not plan["split"] and plan["max_nsplit"] == 1
    rows = plan["plan"]
    assert rows.dtype == torch.int16 and rows.numel() == PLAN_HALFWORDS
    stride = plan["stride"]
    for tile, row in enumerate(_decode_small_rows(plan)):
        head, qb = divmod(tile, 5)
        selected = mask[head, qb].nonzero().flatten().tolist()
        assert row["tile"] == tile and row["nsplit"] == 1 and row["split"] == 0
        assert row["count"] == len(selected) and row["blocks"] == selected
        raw = rows[tile * stride : (tile + 1) * stride].tolist()
        assert all(v == -1 for v in raw[plan["meta_halfwords"] + len(selected) :])
    assert bool((rows[15 * stride :] == -1).all())
    with pytest.raises(ValueError, match="cannot hold"):
        plan_small(mask, kmax=3)
    with pytest.raises(ValueError, match="exceed"):
        plan_small(_random_mask(2, 128, 32, 6, ragged=False))


@pytest.mark.parametrize("kmax", [4, 6])
def test_plan_small_split_slices_cover_each_selection_once(kmax):
    mask = _random_mask(1, 16, 64, 16, seed=5, ragged=True)
    plan = plan_small(mask, kmax=kmax, split=True)
    rows = _decode_small_rows(plan)
    assert plan["split"] and plan["num_items"] == len(rows) and plan["num_tiles"] == 16
    for tile in range(16):
        selected = mask[0, tile].nonzero().flatten().tolist()
        slices = [r for r in rows if r["tile"] == tile]
        nsplit = -(-len(selected) // kmax)
        assert len(slices) == nsplit and all(r["nsplit"] == nsplit for r in slices)
        assert [r["split"] for r in slices] == list(range(nsplit))
        assert all(1 <= r["count"] == len(r["blocks"]) <= kmax for r in slices)
        assert max(r["count"] for r in slices) - min(r["count"] for r in slices) <= 1
        assert sorted(b for r in slices for b in r["blocks"]) == selected
    # Slices of one query block are consecutive items (the merge derives the
    # first item as ``item - split``).
    firsts = [i for i, r in enumerate(rows) if r["split"] == 0]
    assert all(
        rows[i + j]["tile"] == rows[i]["tile"]
        for i in firsts
        for j in range(rows[i]["nsplit"])
    )
    assert plan["max_nsplit"] == max(r["nsplit"] for r in rows) <= MAX_NSPLIT
    with pytest.raises(ValueError, match="slices"):
        plan_small(torch.ones((1, 1, 200), dtype=torch.bool), kmax=4, split=True)


def test_plan_small_split_cluster_rows_are_one_row_per_query_block():
    """A split-cluster plan carries one ``[count, blk...]`` row per query block with ``kmax * cluster`` id slots
    (ascending ids, ``-1`` padded) and launches ``cluster`` items per query block; the kernel derives each slice
    (its cluster rank) and the balanced slice bounds itself, so tiles with at most ``kmax`` blocks leave the
    other ranks idle (``max_nsplit`` counts the real slices)."""
    mask = _random_mask(2, 8, 64, 24, seed=23, ragged=True)
    counts = mask.sum(-1).reshape(-1).tolist()
    for kmax, csize in SPLIT_CLUSTER_VARIANTS:
        if kmax * csize < 24:
            continue
        plan = plan_small(mask, kmax=kmax, split=True, cluster=csize)
        assert plan["split"] and plan["cluster"] == csize and plan["kmax"] == kmax
        assert plan["stride"] == kmax * csize + PLAN_META_UNSPLIT
        assert plan["meta_halfwords"] == PLAN_META_UNSPLIT
        assert plan["num_tiles"] == 16 and plan["num_items"] == 16 * csize
        assert plan["max_nsplit"] == max(-(-c // kmax) for c in counts) <= csize
        rows = _decode_small_rows(plan)
        assert len(rows) == 16
        for tile, row in enumerate(rows):
            selected = mask.reshape(16, 64)[tile].nonzero().flatten().tolist()
            assert row["count"] == len(selected) == counts[tile]
            assert row["blocks"] == selected  # ascending, no padding slot used
            raw = plan["plan"][
                tile * plan["stride"] : (tile + 1) * plan["stride"]
            ].tolist()
            assert all(v == -1 for v in raw[1 + len(selected) :])
        assert bool((plan["plan"][16 * plan["stride"] :] == -1).all())
    # the smallest variant that holds the selection is chosen when kmax is not given
    inferred = plan_small(mask, split=True, cluster=4)
    assert (inferred["kmax"], inferred["cluster"]) == (6, 4)
    with pytest.raises(ValueError, match="cannot hold"):
        plan_small(mask, kmax=6, split=True, cluster=3)
    with pytest.raises(ValueError, match="no split-cluster-3 variant holds"):
        plan_small(mask, split=True, cluster=3)
    with pytest.raises(ValueError, match="exceed"):
        plan_small(
            _random_mask(2, 32, 64, 48, ragged=False), kmax=6, split=True, cluster=8
        )


def _uniform_mask(h, mb, nb, k, seed=42):
    """``k`` selected KV blocks per query block, the same count on every row (a ``-ragged0`` benchmark row)."""
    return _topk_mask(h, mb, nb, k, seed=seed)


def test_small_route_rule_split_cluster_ingest():
    """A split-KV route is upgraded to its split-cluster variant only when one holds the selection, fits the plan,
    places in one wave of clusters and saves at least SPLIT_CLUSTER_MIN_SAVING_KB of per-CTA L2 ingest (the
    longest slice's K/V bytes plus the merging CTA's partial reads): the 16-tile 24-block rows save 64 KB with
    four slices of six (k6sc4), the 8-tile 48-block row 160 KB with eight (k6sc8), and the 32-tile 16-block row
    would read 16 KB more as k6sc3 (one block more per slice), so it stays on k4s."""
    k24 = _uniform_mask(1, 16, 64, 24)  # h1-m1024-n4096-k24
    assert small_route(k24, sms=132, cluster_capacity=CAP) == (6, True, 4)
    assert split_kmax([24] * 16, sms=132) == 4
    assert (
        _split_ingest_kb([24] * 16, 4, None) == 288
        and _split_ingest_kb([24] * 16, 6, 4) == 224
    )
    assert small_route(_uniform_mask(4, 4, 128, 24), sms=132, cluster_capacity=CAP) == (
        6,
        True,
        4,
    )
    k48 = _uniform_mask(1, 8, 128, 48)  # h1-m512-n8192-k48
    assert small_route(k48, sms=132, cluster_capacity=CAP) == (6, True, 8)
    assert split_kmax([48] * 8, sms=132) == 6
    assert (
        _split_ingest_kb([48] * 8, 6, None) == 416
        and _split_ingest_kb([48] * 8, 6, 8) == 256
    )
    k16 = _uniform_mask(
        8, 4, 64, 16
    )  # h8-m256-n4096-k16: k6sc3 would cost 240 KB against 224 KB on k4s
    assert small_route(k16, sms=132, cluster_capacity=CAP) == (4, True, 0)
    assert (
        _split_ingest_kb([16] * 32, 4, None) == 224
        and _split_ingest_kb([16] * 32, 6, 3) == 240
    )
    assert (
        split_cluster_variant_for([16] * 32, split_kmax=4, cluster_capacity=CAP) is None
    )
    # the rule's three refusals: capacity, one wave of clusters, the saving threshold
    assert split_cluster_variant_for([24] * 16, split_kmax=4, cluster_capacity=CAP) == (
        6,
        4,
    )
    assert (
        split_cluster_variant_for([24] * 31, split_kmax=4, cluster_capacity=CAP) is None
    )  # 31 > 30 clusters
    assert (
        split_cluster_variant_for(
            [24] * 16, split_kmax=4, cluster_capacity={2: 66, 3: 39, 6: 17}
        )
        is None
    )
    assert (
        split_cluster_variant_for([48] * 16, split_kmax=6, cluster_capacity=CAP) is None
    )  # 16 > 15 clusters
    assert split_cluster_variant_for([48] * 8, split_kmax=6, cluster_capacity=CAP) == (
        6,
        8,
    )
    assert (
        split_cluster_variant_for([49] * 8, split_kmax=6, cluster_capacity=CAP) is None
    )  # no variant holds 49
    assert SPLIT_CLUSTER_MIN_SAVING_KB == 48 and planner.SPLIT_CLUSTER_ROUTE is True
    # with the route disabled the split-KV variant is kept
    try:
        planner.SPLIT_CLUSTER_ROUTE = False
        assert small_route(k24, sms=132, cluster_capacity=CAP) == (4, True, 0)
        assert small_route(k48, sms=132, cluster_capacity=CAP) == (6, True, 0)
    finally:
        planner.SPLIT_CLUSTER_ROUTE = True
    # the DSM-cluster route still wins where it applies and the unsplit rule comes first
    assert small_route(_uniform_mask(1, 16, 16, 16), sms=132, cluster_capacity=CAP) == (
        4,
        False,
        4,
    )
    assert small_route(_uniform_mask(8, 16, 16, 4), sms=132, cluster_capacity=CAP) == (
        4,
        False,
        0,
    )


def test_small_route_rule_underfilled_persistent_grid():
    """h8-m128-n32768-k64 (16 query blocks of 64 KV blocks) would run 16
    persistent CTAs on 132 SMs and no one-wave split or cluster variant holds 64 blocks, so the route
    takes the KMAX 6 split-KV variant in two waves (176 items; the fewest items of the fitting variants):
    35.2 us -> 29.2 us measured on H100.  h8-m256-n16384-k32 (32 query blocks of 32 blocks) stays on the
    persistent kernel: 32 * SPLIT_UNDERFILL_SMS > 132 (both split variants measured slower there)."""
    k64 = _uniform_mask(8, 2, 512, 64)  # h8-m128-n32768-k64
    assert small_route(k64, sms=132, cluster_capacity=CAP) == (6, True, 0)
    assert split_kmax([64] * 16, sms=132, waves=2) == 6
    assert (
        16 * -(-64 // 6) == 176 > SMALL_OCCUPANCY[6] * 132
    )  # two waves of one CTA per SM
    with pytest.raises(ValueError):
        split_kmax([64] * 16, sms=132)
    k32 = _uniform_mask(8, 4, 256, 32)  # h8-m256-n16384-k32: 32 tiles, 32 * 8 > 132
    assert small_route(k32, sms=132, cluster_capacity=CAP) is None
    assert planner.SPLIT_UNDERFILL_SMS == 8 and planner.SPLIT_UNDERFILL_WAVES == 2
    try:
        planner.SPLIT_UNDERFILL_SMS = 0  # disabled: the persistent kernel as before
        assert small_route(k64, sms=132, cluster_capacity=CAP) is None
    finally:
        planner.SPLIT_UNDERFILL_SMS = 8


def test_split_variants_are_the_shipped_ones():
    """Only the KMAX 4 and 6 split-KV kernels ship: ``split_kmax`` never names
    another variant and ``plan_small`` refuses to plan one.  The retired KMAX 1
    split was the planner's choice only on a band of ragged masks whose grid
    sits at the sliced-route edge (``2 * tiles`` within two of the SM count)
    with one outlier selection; those masks now run the persistent kernel."""
    for counts in ([2], [3] * 8, [7] * 65 + [13], [7] * 64 + [25], [9] * 40):
        assert split_kmax(counts) in (4, 6)
        # no shipped variant fits one wave: small_route falls through
        with contextlib.suppress(ValueError):
            assert split_kmax(counts, sms=132) in (4, 6)
    for kmax in (1, 3):
        with pytest.raises(ValueError, match="split-KV variant"):
            plan_small(_random_mask(1, 4, 8, 6, seed=3), kmax=kmax, split=True)
    # 66 tiles, 65 x 7 blocks + one 13-block selection: KMAX 3/4/6 exceed one wave
    # of 132 SMs, no cluster variant holds 66 tiles -> persistent kernel.
    mask = torch.zeros((6, 11, 16), dtype=torch.bool)
    mask[:, :, :7] = True
    mask[5, 10, :13] = True
    assert small_route(mask, sms=132, cluster_capacity=CAP) is None


def test_stage_table_builds_every_stage_from_the_shared_launcher():
    """Every shipped stage is one ``MODULES`` record naming one ``build.json`` library whose two
    sources are the stage's own kernel and binding units with verified digests; the unshipped
    split variants (``small_k1s``, ``small_k3s`` and their FP8 twins) are refused by name."""
    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == pkg.__name__ == "flashinfer.cake_vsa_sm90_backend"
    libraries = cake_build.libraries_by_name()
    package = Path(cake_build.__file__).parent
    # 4 + 15 BF16 stages, their 19 FP8 twins, 2 quantizers, then the 19 + 19 GQA stages, the
    # two round-2 FP8 GQA programs (attention_fp8_gqa_xs_queue, small_fp8_gqa_k4p2) and the
    # round-16 position-selected FP8 GQA static program (attention_fp8_gqa_pin)
    assert len(loader.STAGES) == 81 and set(loader.STAGES) == set(loader.MODULES)
    referenced = set()
    for stage in loader.STAGES:
        record = loader.stage_record(stage)
        assert record["library"] == record["kernel"], stage
        assert record["kernel_symbol"] == f"kernel_{record['library']}", stage
        assert record["arches"] == ["sm_90a"] and record["ffi_entry"] == "run", stage
        assert not {"sources", "source_sha256", "compile_flags", "cache_names"} & set(
            record
        ), stage
        for arch in record["arches"]:
            library = loader.library_name(stage, arch)
            referenced.add(library)
            entry = libraries[library]
            assert entry["arch"] == arch and entry["entry"] == "run"
            assert entry["sources"] == [
                f"csrc/{record['library']}_kernel.cu",
                f"csrc/{record['library']}_binding.cu",
            ]
            for source in entry["sources"]:
                data = (package / source).read_bytes()
                assert hashlib.sha256(data).hexdigest() == entry["sha256"][source], (
                    source
                )
                assert data.startswith(
                    b"/*\n * Copyright (c) 2026 by FlashInfer team."
                ), source
            binding = (package / entry["sources"][1]).read_text()
            assert record["kernel_symbol"] in binding
            assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(run," in binding
    assert referenced == set(libraries)
    for name in (
        "small_k1s",
        "small_k3s",
        "small_k2",
        "attention_pdl",
        "small_fp8_k1s",
        "small_fp8_k3s",
        "small_fp8_k2",
        "attention_fp8_pdl",
    ):
        with pytest.raises(ValueError, match="unknown Cake SM90 VSA stage"):
            loader.stage_record(name)
    with pytest.raises(ValueError, match="unknown Cake SM90 VSA stage"):
        loader.small_stage(1, split=True)
    with pytest.raises(ValueError, match="unknown Cake SM90 VSA stage"):
        loader.small_stage(3, split=True, fp8=True)
    with pytest.raises(RuntimeError, match="not delivered for"):
        loader.library_name("attention", "sm_100a")


def test_stage_table_geometry_matches_the_stage_names():
    """A record's route, launch record, grid rule, workspace rule and argument plan say what its
    stage name says: persistent stages launch 384 threads with the by-value ``hdr`` and the device
    ``meta`` table; small stages launch one CTA per plan item (128 threads for ``kmax`` 1, else 256),
    a cluster of the name's size, and only the split and split-cluster variants carry the FP32
    workspace buffers.  An FP8 stage (``route["dtype"] == "e4m3"``) is the twin of the BF16 stage of
    the same name with the three scale descriptors ``Qs`` / ``Ks`` / ``Vs`` and the ``k_scale_groups``
    scalar added; a GQA stage (``route["gqa"]``) is the twin of the MHA stage of the same name with
    the small stages taking the ``g_shift`` scalar; the two quantizer stages carry the SM100 Sage
    package's argument plan and a ``batch_heads_chunks`` grid rule the plan loader refuses (they are
    launched with an explicit grid)."""
    scale_logical = {
        "Qs": "q_scale",
        "Ks": "k_scale",
        "Vs": "v_scale",
        "k_scale_groups": "k_scale_groups",
    }
    for stage in loader.STAGES:
        record = loader.stage_record(stage)
        launch, route = record["launch"], record["route"]
        plan = [tuple(pair) for pair in record["arg_plan"]]
        buffers = [name for kind, name in plan if kind == "buffer"]
        assert plan[-3:] == [("grid", "grid_x"), ("grid", "grid_y"), ("grid", "grid_z")]
        assert set(record["logical"]) == {name for kind, name in plan if kind != "grid"}
        assert route == loader.route_of(stage)
        assert launch["cooperative"] is False and launch["use_pdl"] is False
        dtype = loader.stage_dtype(stage)
        if dtype is None:
            assert stage in loader.QUANTIZE_STAGES and route == {"kind": stage}
            assert record["grid_rule"] == {"kind": "batch_heads_chunks"}
            assert record["workspace_rule"] is None
            assert (tuple(launch["block"]), tuple(launch["cluster"])) == (
                (128, 1, 1),
                (1, 1, 1),
            )
            assert 0 <= launch["dynamic_smem_bytes"] <= 232448
            # the quantizers' logical names are their kernel parameter names (SM100 Sage package form)
            assert all(key == value for key, value in record["logical"].items())
            with pytest.raises(ValueError, match="batch_heads_chunks"):
                loader.launch_grid(stage, {"num_ctas": 1, "num_items": 1})
            continue
        fp8 = dtype == "e4m3"
        gqa = loader.stage_gqa(stage)
        assert route.get("gqa", False) is gqa and (("gqa" in route) is gqa)
        assert (("parameter", "k_scale_groups") in plan) is fp8
        for parameter, logical in scale_logical.items():
            assert (record["logical"].get(parameter) == logical) is fp8, (
                stage,
                parameter,
            )
        scale_buffers = ["Qs", "Ks", "Vs"] if fp8 else []
        if route["kind"] == "persistent":
            base = "attention_fp8" if fp8 else "attention"
            if gqa:
                base += "_gqa"
            assert stage.startswith(base)
            # the shared-transposer program is a queue program of the FP8 GQA family (round 2); it states xshare;
            # the position-selected program is a static program of the same family (round 16); it states pin
            xshare = stage == f"{base}_xs_queue"
            pin = stage == f"{base}_pin"
            expected_route = {
                "compact": stage == f"{base}_compact",
                "kind": "persistent",
                "queue": stage == f"{base}_queue" or xshare,
                "tagged": stage == f"{base}_tagged",
            }
            if fp8:
                expected_route["dtype"] = "e4m3"
            if gqa:
                expected_route["gqa"] = True
            if xshare:
                assert fp8 and gqa
                expected_route["xshare"] = True
            if pin:
                assert fp8 and gqa
                expected_route["pin"] = True
            assert route == expected_route
            assert stage == loader.persistent_stage(
                route["queue"],
                route["compact"],
                route["tagged"],
                fp8=fp8,
                gqa=gqa,
                xshare=route.get("xshare", False),
                pin=route.get("pin", False),
            )
            assert ("parameter", "g_shift") not in plan and "g_shift" not in record[
                "logical"
            ]
            # O1 head-pair tiles add no shared memory: the GQA programs state the MHA twins' values
            assert (
                launch["dynamic_smem_bytes"]
                == {
                    (False, False): 232064,
                    (False, True): 232192,
                    (True, False): 185216,
                    (True, True): 185344,
                }[(fp8, route["queue"])]
            )
            assert (tuple(launch["block"]), tuple(launch["cluster"])) == (
                (384, 1, 1),
                (1, 1, 1),
            )
            assert record["grid_rule"] == {"count": "num_ctas", "kind": "plan_count"}
            assert record["workspace_rule"] == {
                "dbg_words": 64,
                "kind": "persistent",
                "tl_words": 8,
            }
            assert ("param_array", "hdr") in plan
            if fp8:
                assert plan.index(("parameter", "k_scale_groups")) + 1 == plan.index(
                    ("buffer", "meta")
                )
            # the FP8 kernels take V (not V^T) and the scales right after O; the BF16 kernels are unchanged
            v_name = "V" if fp8 else "Vt"
            assert buffers == [
                "Q",
                "K",
                v_name,
                "O",
                *scale_buffers,
                "meta",
                "dbg",
                "tl",
            ], stage
        else:
            assert route["kind"] == "small"
            # the depth-2 k4 program states its prefetch depth (round 2); every other small stage states none
            assert stage == loader.small_stage(
                route["kmax"],
                route["split"],
                route["cluster"],
                fp8=fp8,
                gqa=gqa,
                prefetch=route.get("prefetch"),
            )
            assert ("prefetch" in route) is (stage == "small_fp8_gqa_k4p2")
            assert route.get("dtype") == ("e4m3" if fp8 else None)
            # the small GQA stages take the group as the ``g_shift`` scalar (``kv_head = head >> g_shift``)
            assert (("parameter", "g_shift") in plan) is gqa
            assert (record["logical"].get("g_shift") == "g_shift") is gqa
            assert tuple(launch["cluster"]) == (route["cluster"] or 1, 1, 1)
            assert tuple(launch["block"]) == (128 if route["kmax"] == 1 else 256, 1, 1)
            assert record["grid_rule"] == {"count": "num_items", "kind": "plan_count"}
            assert ("param_array", "plan") in plan
            if route["split"]:
                if route["cluster"]:
                    assert (route["kmax"], route["cluster"]) in SPLIT_CLUSTER_VARIANTS
                    assert (
                        stage
                        == f"small_{'fp8_' if fp8 else ''}{'gqa_' if gqa else ''}k{route['kmax']}sc{route['cluster']}"
                    )
                else:
                    assert route["kmax"] in SPLIT_KMAX_VARIANTS
                assert record["workspace_rule"] == {
                    "counter_words": "num_tiles",
                    "item_floats": SMALL_ITEM_ELEMS,
                    "kind": "split",
                    "stats_floats": SMALL_STATS_FLOATS,
                }
                workspace_buffers = ["Wo", "Ws", "Wc"]
            else:
                assert record["workspace_rule"] is None
                workspace_buffers = []
                if route["cluster"]:
                    assert (route["kmax"], route["cluster"]) in CLUSTER_VARIANTS
                else:
                    assert route["kmax"] in SMALL_KMAX_VARIANTS
            v_name = "V" if fp8 else "Vt"
            assert buffers == [
                "Q",
                "K",
                v_name,
                "O",
                *scale_buffers,
                *workspace_buffers,
            ], stage
            if fp8:
                # the k_scale_groups scalar (the per-head stride of Ks) precedes the by-value plan
                assert plan.index(("parameter", "k_scale_groups")) + 1 == plan.index(
                    ("param_array", "plan")
                )
        assert 0 < launch["dynamic_smem_bytes"] <= 232448
        assert loader.launch_grid(stage, {"num_ctas": 7, "num_items": 9}) == (
            (7, 1, 1) if route["kind"] == "persistent" else (9, 1, 1)
        )
    assert set(loader.PERSISTENT_STAGES) == {
        "attention",
        "attention_queue",
        "attention_compact",
        "attention_tagged",
    }
    assert set(loader.SMALL_STAGES) == (
        {loader.small_stage(k) for k in SMALL_KMAX_VARIANTS}
        | {loader.small_stage(k, split=True) for k in SPLIT_KMAX_VARIANTS}
        | {loader.small_stage(k, cluster=c) for k, c in CLUSTER_VARIANTS}
        | {
            loader.small_stage(k, split=True, cluster=c)
            for k, c in SPLIT_CLUSTER_VARIANTS
        }
    )
    assert [loader.small_stage(k, True, c) for k, c in SPLIT_CLUSTER_VARIANTS] == [
        "small_k6sc3",
        "small_k6sc4",
        "small_k6sc8",
    ]
    for stage in ("small_k6sc3", "small_k6sc4", "small_k6sc8"):
        launch = loader.stage_record(stage)["launch"]
        # no receive buffers: the split-cluster kernels compute the split kernel's shared memory
        assert (
            launch["dynamic_smem_bytes"]
            == loader.stage_record("small_k6s")["launch"]["dynamic_smem_bytes"]
        )
        assert tuple(launch["cluster"]) == (int(stage.rsplit("c", 1)[1]), 1, 1)
    with pytest.raises(ValueError, match="no num_items to launch"):
        loader.launch_grid("small_k1", {"num_ctas": 1, "num_items": 0})
    # The FP8 twins: one per BF16 stage, named by the same rule; the quantizers close the table.
    assert set(loader.FP8_PERSISTENT_STAGES) == {
        "attention_fp8",
        "attention_fp8_queue",
        "attention_fp8_compact",
        "attention_fp8_tagged",
    }
    assert set(loader.FP8_SMALL_STAGES) == {
        stage.replace("small_", "small_fp8_") for stage in loader.SMALL_STAGES
    }
    assert loader.QUANTIZE_STAGES == ("quantize_qk_vamax", "quantize_v")
    # The GQA stages: one per MHA attention stage of each data type, named by the same rule with ``gqa``.
    assert set(loader.GQA_PERSISTENT_STAGES) == {
        "attention_gqa",
        "attention_gqa_queue",
        "attention_gqa_compact",
        "attention_gqa_tagged",
    }
    assert set(loader.GQA_SMALL_STAGES) == {
        stage.replace("small_", "small_gqa_") for stage in loader.SMALL_STAGES
    }
    # ... plus the shared-transposer queue stage and the depth-2 grouped-query FP8 k4 stage,
    # the two GQA stages without an MHA twin of their own keywords (round 2), and the
    # position-selected static stage (round 16).
    assert set(loader.FP8_GQA_PERSISTENT_STAGES) == {
        "attention_fp8_gqa",
        "attention_fp8_gqa_queue",
        "attention_fp8_gqa_compact",
        "attention_fp8_gqa_tagged",
        "attention_fp8_gqa_xs_queue",
        "attention_fp8_gqa_pin",
    }
    assert set(loader.FP8_GQA_SMALL_STAGES) == {
        stage.replace("small_", "small_fp8_gqa_") for stage in loader.SMALL_STAGES
    } | {"small_fp8_gqa_k4p2"}
    assert len(loader.PERSISTENT_STAGES) + len(loader.SMALL_STAGES) == 19
    assert len(loader.FP8_PERSISTENT_STAGES) + len(loader.FP8_SMALL_STAGES) == 19
    assert len(loader.GQA_PERSISTENT_STAGES) + len(loader.GQA_SMALL_STAGES) == 19
    assert (
        len(loader.FP8_GQA_PERSISTENT_STAGES) + len(loader.FP8_GQA_SMALL_STAGES) == 22
    )
    assert set(loader.STAGES) == (
        set(loader.PERSISTENT_STAGES)
        | set(loader.SMALL_STAGES)
        | set(loader.FP8_PERSISTENT_STAGES)
        | set(loader.FP8_SMALL_STAGES)
        | set(loader.GQA_PERSISTENT_STAGES)
        | set(loader.GQA_SMALL_STAGES)
        | set(loader.FP8_GQA_PERSISTENT_STAGES)
        | set(loader.FP8_GQA_SMALL_STAGES)
        | set(loader.QUANTIZE_STAGES)
    )
    # registry order: the 40 MHA records and quantizers first, the GQA records appended after them
    assert loader.STAGES[:40] == tuple(
        loader.PERSISTENT_STAGES
        + loader.SMALL_STAGES
        + loader.FP8_PERSISTENT_STAGES
        + loader.FP8_SMALL_STAGES
        + loader.QUANTIZE_STAGES
    )
    # (the two round-2 programs and the round-16 program appended after the 38 GQA records, in
    # this order)
    appended = (
        "attention_fp8_gqa_xs_queue",
        "small_fp8_gqa_k4p2",
        "attention_fp8_gqa_pin",
    )
    assert loader.STAGES[40:78] == tuple(
        stage
        for stage in loader.GQA_PERSISTENT_STAGES
        + loader.GQA_SMALL_STAGES
        + loader.FP8_GQA_PERSISTENT_STAGES
        + loader.FP8_GQA_SMALL_STAGES
        if stage not in appended
    )
    assert loader.STAGES[78:] == appended
    for bf16 in loader.PERSISTENT_STAGES + loader.SMALL_STAGES:
        fp8 = (
            bf16.replace("attention", "attention_fp8")
            if bf16.startswith("attention")
            else bf16.replace("small_", "small_fp8_")
        )
        assert loader.stage_dtype(bf16) == "bf16" and loader.stage_dtype(fp8) == "e4m3"
        a, b = loader.stage_record(bf16), loader.stage_record(fp8)
        assert {k: v for k, v in a["route"].items()} == {
            k: v for k, v in b["route"].items() if k != "dtype"
        }
        assert (
            a["workspace_rule"] == b["workspace_rule"]
            and a["grid_rule"] == b["grid_rule"]
        )
        assert (
            a["launch"]["block"] == b["launch"]["block"]
            and a["launch"]["cluster"] == b["launch"]["cluster"]
        )
        assert a["library"] != b["library"], "an FP8 program is another program"
    for mha in (
        loader.PERSISTENT_STAGES
        + loader.SMALL_STAGES
        + loader.FP8_PERSISTENT_STAGES
        + loader.FP8_SMALL_STAGES
    ):
        gqa = _gqa_stage_of(mha)
        assert not loader.stage_gqa(mha) and loader.stage_gqa(gqa), (mha, gqa)
        assert loader.stage_dtype(gqa) == loader.stage_dtype(mha)
        a, b = loader.stage_record(mha), loader.stage_record(gqa)
        assert b["route"] == {**a["route"], "gqa": True}
        assert (
            a["workspace_rule"] == b["workspace_rule"]
            and a["grid_rule"] == b["grid_rule"]
        )
        assert (
            a["launch"]["block"] == b["launch"]["block"]
            and a["launch"]["cluster"] == b["launch"]["cluster"]
        )
        assert a["library"] != b["library"], "a GQA program is another program"
        if a["route"]["kind"] == "small":
            # the same kernel parameters plus the g_shift scalar
            assert [
                tuple(x) for x in b["arg_plan"] if tuple(x) != ("parameter", "g_shift")
            ] == [tuple(x) for x in a["arg_plan"]]
            assert {k: v for k, v in b["logical"].items() if k != "g_shift"} == a[
                "logical"
            ]
            assert (
                b["launch"]["dynamic_smem_bytes"] == a["launch"]["dynamic_smem_bytes"]
            )
        else:
            assert b["arg_plan"] == a["arg_plan"] and b["logical"] == a["logical"]
    assert "g_shift" in loader.LOGICAL_SCALARS
    # The depth-2 grouped-query FP8 k4 stage: the ``small_fp8_gqa_k4`` record with ``prefetch`` in its route and
    # its own program (the depth only changes when blocks 2 / 3 are loaded; same parameters, same footprint).
    a, b = (
        loader.stage_record("small_fp8_gqa_k4"),
        loader.stage_record("small_fp8_gqa_k4p2"),
    )
    assert b["route"] == {**a["route"], "prefetch": 2} and loader.stage_gqa(
        "small_fp8_gqa_k4p2"
    )
    assert loader.stage_dtype("small_fp8_gqa_k4p2") == "e4m3"
    assert b["arg_plan"] == a["arg_plan"] and b["logical"] == a["logical"]
    assert b["launch"] == a["launch"] and b["grid_rule"] == a["grid_rule"]
    assert b["workspace_rule"] is None and b["library"] != a["library"]
    assert loader.STAGES[-2:] == ("small_fp8_gqa_k4p2", "attention_fp8_gqa_pin")
    assert loader.small_stage(4, fp8=True, gqa=True, prefetch=2) == "small_fp8_gqa_k4p2"
    assert loader.small_stage(4, fp8=True, gqa=True) == "small_fp8_gqa_k4"
    with pytest.raises(ValueError, match="unknown Cake SM90 VSA stage"):
        loader.small_stage(4, fp8=True, prefetch=2)  # no MHA stage at depth 2
    with pytest.raises(ValueError, match="unknown Cake SM90 VSA stage"):
        loader.small_stage(4, gqa=True, prefetch=2)  # no BF16 stage at depth 2
    # The shared-transposer queue stage: the ``attention_fp8_gqa_queue`` record with ``xshare`` in
    # its route and its own program (the V^T transposition is dealt to three producer warps; same
    # parameters, same footprint).
    a, b = (
        loader.stage_record("attention_fp8_gqa_queue"),
        loader.stage_record("attention_fp8_gqa_xs_queue"),
    )
    assert b["route"] == {**a["route"], "xshare": True} and loader.stage_gqa(
        "attention_fp8_gqa_xs_queue"
    )
    assert loader.stage_dtype("attention_fp8_gqa_xs_queue") == "e4m3"
    assert b["arg_plan"] == a["arg_plan"] and b["logical"] == a["logical"]
    assert b["launch"] == a["launch"] and b["grid_rule"] == a["grid_rule"]
    assert b["workspace_rule"] == a["workspace_rule"] and b["library"] != a["library"]
    assert (
        loader.persistent_stage(True, fp8=True, gqa=True, xshare=True)
        == "attention_fp8_gqa_xs_queue"
    )
    assert (
        loader.persistent_stage(True, fp8=True, gqa=True) == "attention_fp8_gqa_queue"
    )


def _gqa_stage_of(stage):
    """The GQA stage serving the route of MHA stage ``stage`` (``attention`` -> ``attention_gqa``,
    ``small_fp8_k4s`` -> ``small_fp8_gqa_k4s``)."""
    if stage.startswith("attention_fp8"):
        return stage.replace("attention_fp8", "attention_fp8_gqa", 1)
    if stage.startswith("attention"):
        return stage.replace("attention", "attention_gqa", 1)
    if stage.startswith("small_fp8_"):
        return stage.replace("small_fp8_", "small_fp8_gqa_", 1)
    return stage.replace("small_", "small_gqa_", 1)


def test_build_manifest_pins_the_numerics_flags():
    """The flag contract: every entry compiles with ``--use_fast_math`` (the flag the kernels were
    validated and measured with) and nothing else; no ``--fmad`` flag is recorded (nvcc's default
    contraction applies), the common flags carry no numerics flag, and the loader lets the manifest
    rather than itself state fast math (``use_fast_math=False``)."""
    manifest = cake_build.manifest()
    assert manifest["arch_flags"] == {
        "sm_90a": ["-gencode=arch=compute_90a,code=sm_90a"]
    }
    assert manifest["common"]["requires"]["cuda_toolkit"]["min"] == "12.8"
    numerics = (
        "--fmad",
        "--use_fast_math",
        "-use_fast_math",
        "--ftz",
        "--prec-div",
        "--prec-sqrt",
    )
    assert not any(
        flag.startswith(numerics) for flag in manifest["common"]["nvcc_flags"]
    )
    assert len(manifest["libraries"]) == 81
    for entry in manifest["libraries"]:
        assert entry["nvcc_flags"] == ["--use_fast_math"], entry["name"]
        assert entry["nvcc_flags_if"] == [], entry["name"]
        assert entry["min_cuda"] == "12.8", entry["name"]
    assert "use_fast_math=False" in inspect.getsource(cake_build.jit_spec)


def test_package_imports_only_its_allowlist():
    """The package imports itself, ``torch``, ``tvm_ffi`` and ``cuda.bindings`` (the cluster-capacity
    probe) beside the standard library; the pinned loader reaches ``flashinfer.jit`` lazily on its JIT
    path only; nothing reads the environment."""
    package = Path(pkg.__file__).parent
    allowed = {
        "__future__",
        "functools",
        "hashlib",
        "heapq",
        "json",
        "math",
        "pathlib",
        "re",
        "typing",
    }
    allowed |= {"torch", "tvm_ffi", "cuda"}
    for path in sorted(package.glob("*.py")):
        source = path.read_text()
        assert "os.environ" not in source and "import os" not in source, path.name
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                names = [node.module]
            else:
                continue
            for name in names:
                root = name.split(".")[0]
                if path.name == "cake_build.py" and root == "flashinfer":
                    continue  # the pinned loader's lazy JIT path
                assert root in allowed, (path.name, name)
    # The planner's probe reaches the driver through cuda.bindings only (no flashinfer.cuda_utils).
    assert "cuda_utils" not in Path(planner.__file__).read_text()


@pytest.fixture(scope="module", autouse=True)
def _warm_jit():
    """Build every ``sm_90a`` library once (FlashInfer's parallel ``build_jit_specs``) so that the
    first plan of each stage does not pay a serial JIT compile."""
    if torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0):
        arch = loader.arch_for_capability(torch.cuda.get_device_capability())
        specs = loader.specs_for_arch(arch)
        if specs:
            from flashinfer.jit import build_jit_specs

            build_jit_specs(specs, verbose=False)
    yield


def _tile_rows(plan):
    """Per-CTA plan rows: the static rows ``c * stride + i`` (the first row's
    META_NTILES word = static count), plus, in the queue layout, a round-robin
    preview of the pool rows (from ``g * stride``, handed out at run time) and
    a check of the all-zero sentinel rows that follow them."""
    meta = plan["meta"]
    g, stride, n_pool = plan["num_ctas"], plan["tile_stride"], len(plan["pool"])
    out = []
    for c in range(g):
        n_static = int(meta[c * stride, META_NTILES])
        assert 1 <= n_static <= stride
        assert all(
            int(meta[c * stride + i, META_NTILES]) == int(plan["queue"])
            for i in range(1, n_static)
        )
        out.append([c * stride + i for i in range(n_static)])
    if not plan["queue"]:
        assert n_pool == 0 and meta.shape[0] == plan["num_rows"] == g * stride
        return out
    assert n_pool > 0 and meta.shape[0] == plan["num_rows"] == g * stride + n_pool + g
    assert all(int(meta[g * stride + j, META_NTILES]) == 1 for j in range(n_pool))
    assert not meta[g * stride + n_pool :].any(), "sentinel rows must be all zero"
    for c in range(g):
        out[c].extend(g * stride + j for j in range(c, n_pool, g))
    return out


def _decode_tiles(plan):
    """Yield (head, qb0, qb1, mode, positions[(blk_a, blk_b)], owns[2]) per tile."""
    meta = plan["meta"].numpy().view("uint32")
    for rows in _tile_rows(plan):
        for r in rows:
            row = meta[r]
            head, qb0, qb1, mode = (int(x) for x in row[:4])
            n_seq = int(row[META_NSEQ])
            words = row[META_SEQ_OFF : META_SEQ_OFF + n_seq]
            positions = []
            for w in words:
                a, b = int(w & 0xFFFF), int(w >> 16)
                positions.append((a, b if b != 0xFFFF else -1))
            owns = []
            for wg in range(2):
                n_own = int(row[META_NOWN + wg])
                assert n_own <= MAX_OWN
                base = META_OWN_OFF + wg * OWN_WORDS
                packed = row[base : base + (n_own + 1) // 2]
                entries = []
                for w in packed:
                    entries.extend((int(w & 0xFFFF), int(w >> 16)))
                owns.append(entries[:n_own])
            yield head, qb0, qb1, mode, positions, owns


def _disjoint_mask(h, mb, k):
    """Query block ``qb`` of every head selects blocks ``qb * k .. qb * k + k - 1``: no (head, block) is read twice."""
    mask = torch.zeros((h, mb, mb * k), dtype=torch.bool)
    for qb in range(mb):
        mask[:, qb, qb * k : (qb + 1) * k] = True
    return mask


def _hint_words(plan):
    hdr = plan["hdr"].view(planner.HDR_CTAS, planner.HDR_WORDS)
    assert int(hdr[plan["num_ctas"] :].abs().sum()) == 0, "unused CTA headers stay zero"
    return hdr[: plan["num_ctas"], planner.HDR_KV_HINT]


def test_kv_hint_rule_no_reuse_plan_is_evict_first():
    plan = plan_vsa_sm90(_disjoint_mask(2, 8, 4), sms=132)
    assert plan["kv_hint"] == 1
    assert bool((_hint_words(plan) == 1).all())


def test_kv_hint_rule_full_reuse_plan_is_evict_last():
    # every query block of a head selects the same 8 blocks: touched = 8 per head, streamed >= 8 x 8 per head
    mask = torch.zeros((2, 16, 32), dtype=torch.bool)
    mask[:, :, :8] = True
    plan = plan_vsa_sm90(mask, sms=132)
    assert plan["kv_hint"] == 0
    assert bool((_hint_words(plan) == 0).all())


def test_kv_hint_rule_threshold():
    # streamed / touched <= 1.25 -> evict_first (1); above -> evict_last (0); the pair-tile layout lists two blocks
    # per position, -1 marks an empty slot
    mask = torch.zeros((1, 1, 4), dtype=torch.bool)
    mask[0, 0, :4] = True  # touched = 4
    assert (
        planner.kv_l2_hint(mask, [[0, 1, 2, 3, 0]]) == 1
    )  # 5 streamed, 5 * 4 <= 5 * 4
    assert (
        planner.kv_l2_hint(mask, [[0, 1, 2, 3, 0, 1]]) == 0
    )  # 6 streamed, 6 * 4 > 5 * 4
    assert (
        planner.kv_l2_hint(mask, [[0, 1, -1, 2, 3, -1]]) == 1
    )  # empty slots do not stream
    assert planner.HDR_KV_HINT == 11 and planner.HDR_WORDS == 12


TAG = 1 << planner.KV_TAG_BIT


def _block_ids(plan):
    """(head, raw int16 id) of every block slot of every tile row; -1 = no second block."""
    meta, out = plan["meta"], []
    for rows in _tile_rows(plan):
        for r in rows:
            head, n = int(meta[r, 0]), int(meta[r, META_NSEQ])
            for pos in range(n):
                w = int(meta[r, META_SEQ_OFF + pos]) & 0xFFFFFFFF
                for half in (w & 0xFFFF, w >> 16):
                    out.append((head, half - 0x10000 if half >= 0x8000 else half))
    return out


def _reader_counts(ids):
    readers = {}
    for head, blk in ids:
        if blk >= 0:
            key = (head, blk & ~TAG)
            readers[key] = readers.get(key, 0) + 1
    return readers


@pytest.mark.parametrize(
    "h,mb,nb,k,tagged",
    [
        (
            32,
            8,
            128,
            16,
            True,
        ),  # h32-m512-n8192-k16: 84.5 MiB footprint, 34.0 MiB re-read
        (16, 16, 128, 12, True),  # h16-m1024-n8192-k12: 51.1 / 28.8 MiB
        (
            8,
            16,
            512,
            64,
            False,
        ),  # h8-m1024-n32768-k64: 112.8 / 78.5 MiB, re-reads do not fit
        (8, 8, 128, 16, False),  # ~21 MiB with single-reader blocks: resident untagged
        (8, 32, 32, 16, False),  # h8-m2048-n2048-k16: no single-reader block
    ],
)
def test_single_reader_tags_follow_the_l2_fit_rule(h, mb, nb, k, tagged):
    """On an evict_last static launch the planner tags the blocks exactly one tile loads
    (bit KV_TAG_BIT of the id, bit 1 of HDR_KV_HINT) only when the launch's K + V^T
    footprint exceeds KV_TAG_L2_BYTES while the re-read blocks alone fit it; every other
    launch carries the shipped rows (the kernel honours the tags only behind the flag)."""
    assert (
        planner.KV_TAG_SINGLE_READER,
        planner.KV_TAG_BIT,
        planner.HDR_KV_TAGGED,
    ) == (
        True,
        14,
        2,
    )
    assert (planner.KV_TAG_L2_BYTES, planner.KV_BLOCK_BYTES) == (40 << 20, 32768)
    plan = plan_vsa_sm90(_topk_mask(h, mb, nb, k), sms=132)
    assert plan["kv_hint"] == 0 and not plan["queue"] and not plan["compact"]
    assert plan["tagged"] is tagged
    ids = _block_ids(plan)
    readers = _reader_counts(ids)
    footprint = len(readers) * planner.KV_BLOCK_BYTES
    reread = sum(1 for n in readers.values() if n > 1) * planner.KV_BLOCK_BYTES
    fits = footprint > planner.KV_TAG_L2_BYTES and reread <= planner.KV_TAG_L2_BYTES
    assert fits is tagged and planner.kv_tags_pay(readers) is tagged, (
        footprint,
        reread,
    )
    want = planner.HDR_KV_TAGGED if tagged else 0
    assert set(int(w) for w in _hint_words(plan)) == {want}
    tagged_ids = {(head, blk & ~TAG) for head, blk in ids if blk >= 0 and blk & TAG}
    single = {key for key, n in readers.items() if n == 1}
    assert tagged_ids == (single if tagged else set())
    assert all(0 <= (blk & ~TAG) < nb for _, blk in ids if blk >= 0)
    hdr = plan["hdr"].view(planner.HDR_CTAS, planner.HDR_WORDS)
    for c in range(plan["num_ctas"]):  # header positions carry the same (tagged) ids
        row = c * plan["tile_stride"]
        for pos in range(int(hdr[c, 4])):
            w = int(plan["meta"][row, META_SEQ_OFF + pos]) & 0xFFFFFFFF
            assert int(hdr[c, 5 + 2 * pos]) & 0xFFFF == w & 0xFFFF
            assert int(hdr[c, 6 + 2 * pos]) & 0xFFFF == w >> 16


def test_tags_off_restores_the_shipped_rows(monkeypatch):
    mask = _topk_mask(32, 8, 128, 16)
    on = plan_vsa_sm90(mask, sms=132)
    assert on["tagged"]
    monkeypatch.setattr(planner, "KV_TAG_SINGLE_READER", False)
    off = plan_vsa_sm90(mask, sms=132)
    assert not off["tagged"] and set(int(w) for w in _hint_words(off)) == {0}
    assert all(not (blk & TAG) for _, blk in _block_ids(off) if blk >= 0)
    stripped = on["meta"].clone()
    seq = stripped[:, META_SEQ_OFF : META_SEQ_OFF + planner.META_SEQ_WORDS]
    seq = seq.to(torch.int64) & 0xFFFFFFFF
    lo, hi = seq & 0xFFFF, seq >> 16
    lo = torch.where(lo == 0xFFFF, lo, lo & ~TAG)
    hi = torch.where(hi == 0xFFFF, hi, hi & ~TAG)
    seq = lo | (hi << 16)
    seq = torch.where(seq >= 2**31, seq - 2**32, seq).to(torch.int32)
    stripped[:, META_SEQ_OFF : META_SEQ_OFF + planner.META_SEQ_WORDS] = seq
    assert torch.equal(stripped, off["meta"])
    hdr_on = on["hdr"].view(planner.HDR_CTAS, planner.HDR_WORDS).to(torch.int64).clone()
    hdr_on[:, 5:11] = torch.where(
        hdr_on[:, 5:11] >= 0, hdr_on[:, 5:11] & ~TAG, hdr_on[:, 5:11]
    )
    hdr_on[:, planner.HDR_KV_HINT] &= ~planner.HDR_KV_TAGGED
    assert torch.equal(hdr_on.to(torch.int16).reshape(-1), off["hdr"].reshape(-1))


@pytest.mark.parametrize(
    "h,mb,nb,capacity,ragged",
    [
        (1, 1, 1, 1, False),
        (2, 3, 5, 2, True),
        (4, 16, 8, 4, True),
        (8, 64, 64, 32, False),
        (7, 64, 64, 16, True),
    ],
)
def test_plan_covers_every_query_block_once_and_is_deadlock_free(
    h, mb, nb, capacity, ragged
):
    mask = _random_mask(h, mb, nb, capacity, ragged=ragged)
    plan = plan_vsa_sm90(mask, sms=132)
    assert plan["meta"].shape[1] == META_WORDS
    # the grid is the full machine or a uniform grid keeping >= 90 % of it busy
    assert (
        0.9 * min(plan["num_tiles"], 132)
        <= plan["num_ctas"]
        <= min(plan["num_tiles"], 132)
    )
    per_cta = [len(t) for t in _cta_tiles(plan)]
    if plan["num_ctas"] < min(plan["num_tiles"], 132):
        assert max(per_cta) - min(per_cta) == 0
    seen = set()
    for head, qb0, qb1, mode, positions, owns in _decode_tiles(plan):
        blocks = {qb0} if mode else {qb0, qb1}
        assert mode in (0, 1, 2)
        for qb in blocks:
            assert (head, qb) not in seen
            seen.add((head, qb))
        # Every selected KV block of each covered query block travels exactly once.
        bits = [0] * len(positions)
        served = {qb: [] for qb in blocks}
        for wg, entries in enumerate(owns):
            for entry in entries:
                pos, has2, use = entry >> 3, (entry >> 2) & 1, entry & 3
                assert use & (1 << wg)
                assert has2 == (positions[pos][1] >= 0)
                bits[pos] |= use
        for pos, (a, b) in enumerate(positions):
            for blk in (a, b):
                if blk >= 0:
                    consumers = (
                        [qb0, qb1]
                        if (mode == 0 and bits[pos] == 3)
                        else ([qb0] if (mode or bits[pos] == 1) else [qb1])
                    )
                    for qb in consumers:
                        served[qb].append(blk)
        for qb in blocks:
            expected = mask[head, qb].nonzero().flatten().tolist()
            assert sorted(served[qb]) == expected, (head, qb)
        assert _schedule_feasible([bt for bt in bits])
    assert seen == {(head, qb) for head in range(h) for qb in range(mb)}


def test_plan_rejects_unsupported_masks():
    with pytest.raises(ValueError, match="empty"):
        mask = torch.ones((1, 2, 3), dtype=torch.bool)
        mask[0, 1] = False
        plan_vsa_sm90(mask, sms=132)
    with pytest.raises(ValueError, match="at most 64"):
        plan_vsa_sm90(torch.ones((1, 1, 65), dtype=torch.bool), sms=132)
    with pytest.raises(ValueError, match="boolean"):
        plan_vsa_sm90(torch.ones((1, 1, 2), dtype=torch.int32), sms=132)


# --------------------------------------------------------------------------- GQA plans (CPU)

GQA_GROUPS = planner.GQA_GROUPS
MODE_HEAD_PAIR = planner.MODE_HEAD_PAIR
MODE_HEAD_PAIR_SPLIT = planner.MODE_HEAD_PAIR_SPLIT
HDR_WORDS = planner.HDR_WORDS


def _replicated(tensor, g, dim=0):
    """The kv-head tensor replicated per group: what the MHA route sees on a GQA problem."""
    return tensor.repeat_interleave(g, dim=dim)


def _gqa_tile_count(g, h_kv, mb, mode, repaired=False):
    """Tiles of a GQA persistent plan: ``h_kv * mb * g / 2`` head-pair tiles (pair mode) or re-paired split tiles
    (split mode under ``repair_split``), ``h_kv * mb * g`` per-head split tiles (split mode, mode-1 replay)."""
    return h_kv * mb * (g // 2 if (mode == "pair" or repaired) else g)


def _chains_of_mha_plan(plan):
    """``(head, qb)`` -> what the MHA plan does for that query block: ``("chain", positions, has_second)`` of
    the warpgroup that serves it in a pair / lone tile (own-list order), ``("split", positions, owns)`` for a
    split tile (both warpgroups, the tile verbatim)."""
    chains = {}
    for head, qb0, qb1, mode, positions, owns in _decode_tiles(plan):
        if mode == 1:
            chains[(head, qb0)] = ("split", positions, owns)
            continue
        blocks = [qb0, qb1] if mode == 0 else [qb0]
        for block, entries in zip(blocks, owns[: len(blocks)], strict=True):
            chain = [positions[e >> 3] for e in entries]
            has2 = [(e >> 2) & 1 for e in entries]
            chains[(head, block)] = ("chain", chain, has2)
    return chains


@pytest.mark.parametrize("g", [2, 4, 8])
@pytest.mark.parametrize(
    "h_kv,mb,nb,capacity,ragged",
    [
        (1, 1, 1, 1, False),  # MQA, one tile
        (2, 3, 5, 2, True),  # tails, odd block counts
        (3, 7, 64, 64, False),  # full 64-block lists: MAX_OWN chains
        (2, 33, 64, 64, True),  # odd query-block count: the lone block of a pair plan
        (4, 16, 8, 4, True),
        (2, 129, 16, 8, True),
        (
            2,
            528,
            64,
            8,
            False,
        ),  # 2112 x g / 2 ... tiles per data type: the queue layout
    ],
)
def test_gqa_plan_replays_the_mha_plan_of_the_replicated_mask(
    g, h_kv, mb, nb, capacity, ragged
):
    """A GQA plan is the MHA plan of the mask replicated per group, replayed: a pair-mode plan as head-pair
    tiles (mode 3; warpgroup ``w`` computes query head ``head + w`` at ``qb0``; both own lists hold every
    position with use bits 3 in the order of the MHA warpgroup's own list for that block), a split-mode plan
    as per-head mode-1 tiles (the MHA tile verbatim) -- or, when the planner of record re-pairs it
    (``repair_split`` "auto": a uniform plan whose mode-1 tiles need more than one wave), as re-paired tiles
    (mode 4; warpgroup ``w`` computes query head ``head + w`` over the MHA split tile's WHOLE position list).
    Plan word 2 (the MHA ``qb1`` slot) and header halfword 2 carry the kv head; the mode is the replicated
    plan's; ``H`` is the query head count."""
    mask = _random_mask(h_kv, mb, nb, capacity, seed=h_kv * 100 + mb, ragged=ragged)
    gqa = plan_vsa_sm90(mask, sms=132, gqa=g)
    mha = plan_vsa_sm90(_replicated(mask, g), sms=132)
    assert (gqa["gqa"], gqa["H_kv"], gqa["H"]) == (g, h_kv, g * h_kv)
    assert gqa["mode"] == mha["mode"] and (gqa["MB"], gqa["NB"]) == (mb, nb)
    repaired = bool(gqa["repair_split"])
    assert repaired == (gqa["mode"] == "split" and not ragged and g * h_kv * mb > 132)
    assert (gqa["repair_rule"], mha["repair_split"]) == ("auto", False)
    # one head-pair tile per (kv head, query block, head pair) in pair mode -- a lone block of the MHA
    # plan is a chain like any other, so the count is not the replicated plan's g x (pairs + lone) --
    # one tile per (query head, query block) in split mode, one per (head pair, query block) re-paired
    assert gqa["num_tiles"] == _gqa_tile_count(g, h_kv, mb, gqa["mode"], repaired)
    assert gqa["meta"].shape[1] == META_WORDS and gqa["hdr"].shape == mha["hdr"].shape
    chains = _chains_of_mha_plan(mha)
    seen = {}
    for head, qb0, word2, mode, positions, owns in _decode_tiles(gqa):
        assert word2 == head // g, "plan word 2 is the kv head"
        if gqa["mode"] == "pair":
            assert mode == MODE_HEAD_PAIR and head % 2 == 0
            assert owns[0] == owns[1] and all(e & 3 == 3 for e in owns[0])
            assert [e >> 3 for e in owns[0]] == list(range(len(positions)))
            assert 0 < len(positions) <= MAX_OWN
            for w in range(2):
                kind, chain, has2 = chains[(head + w, qb0)]
                assert kind == "chain", (head + w, qb0)
                assert positions == chain, (head + w, qb0)
                assert [(e >> 2) & 1 for e in owns[0]] == has2
                seen[(head + w, qb0)] = seen.get((head + w, qb0), 0) + 1
        elif repaired:
            assert mode == MODE_HEAD_PAIR_SPLIT and head % 2 == 0
            assert owns[0] == owns[1] and all(e & 3 == 3 for e in owns[0])
            assert [e >> 3 for e in owns[0]] == list(range(len(positions)))
            assert [(e >> 2) & 1 for e in owns[0]] == [
                int(b >= 0) for _, b in positions
            ]
            kind, mha_positions, _mha_owns = chains[(head, qb0)]
            assert kind == "split" and positions == mha_positions, (head, qb0)
            for w in range(2):
                seen[(head + w, qb0)] = seen.get((head + w, qb0), 0) + 1
        else:
            assert mode == 1
            kind, mha_positions, mha_owns = chains[(head, qb0)]
            assert kind == "split" and positions == mha_positions and owns == mha_owns
            seen[(head, qb0)] = seen.get((head, qb0), 0) + 1
    assert seen == {(head, qb): 1 for head in range(g * h_kv) for qb in range(mb)}
    # the by-value header of every CTA: [head, qb0, kv head, mode, ...] of its first tile
    hdr = gqa["hdr"].tolist()
    meta = gqa["meta"]
    for c in range(gqa["num_ctas"]):
        first = meta[c * gqa["tile_stride"]].tolist()
        assert hdr[c * HDR_WORDS : c * HDR_WORDS + 4] == first[:4], c
        assert hdr[c * HDR_WORDS + 2] == first[0] // g, c
    assert 1 <= gqa["num_ctas"] <= min(gqa["num_tiles"], 132)


@pytest.mark.parametrize("rule", [False, True, "auto"], ids=["off", "on", "auto"])
def test_gqa_split_plans_re_pair_uniform_multi_wave_plans_by_default(rule):
    """``GQA_REPAIR_SPLIT = "auto"``: a uniform split-mode GQA plan whose mode-1 tiles need more than one wave of
    the SMs (``H x MB > sms``) is re-paired by default -- byte for byte the ``repair_split=True`` plan, mode-4
    tiles only, half the tiles, each tile's position list the MHA split tile's whole list -- while
    ``repair_split=False`` keeps the mode-1 replay.  One-wave uniform plans, ragged plans and pair plans are not
    re-paired under "auto" (their plan equals the ``False`` plan byte for byte); other values are rejected."""
    assert planner.GQA_REPAIR_SPLIT == "auto"
    multi = _random_mask(
        2, 32, 32, 8, seed=11, ragged=False
    )  # h8-m2048-n2048-k8 at g = 4: 256 mode-1 tiles
    one_wave = _random_mask(
        2, 16, 16, 12, seed=12, ragged=False
    )  # h8-m1024-n1024-k12 at g = 4: 128 tiles
    ragged = _random_mask(2, 32, 32, 8, seed=13, ragged=True)
    for mask, expect_auto in ((multi, True), (one_wave, False), (ragged, False)):
        plan = plan_vsa_sm90(mask, sms=132, gqa=4, mode="split", repair_split=rule)
        off = plan_vsa_sm90(mask, sms=132, gqa=4, mode="split", repair_split=False)
        expect = rule is True or (rule == "auto" and expect_auto)
        assert (plan["repair_split"], plan["repair_rule"], plan["mode"]) == (
            expect,
            rule,
            "split",
        )
        assert plan["num_tiles"] == _gqa_tile_count(
            4, 2, mask.shape[1], "split", expect
        )
        modes = {mode for _, _, _, mode, _, _ in _decode_tiles(plan)}
        assert modes == ({MODE_HEAD_PAIR_SPLIT} if expect else {planner.MODE_SPLIT})
        if rule == "auto":
            default = plan_vsa_sm90(mask, sms=132, gqa=4, mode="split")
            assert torch.equal(default["meta"], plan["meta"]) and torch.equal(
                default["hdr"], plan["hdr"]
            )
            assert default["repair_split"] == expect
        if not expect:
            assert torch.equal(plan["meta"], off["meta"]) and torch.equal(
                plan["hdr"], off["hdr"]
            )
        else:
            forced = plan_vsa_sm90(
                mask, sms=132, gqa=4, mode="split", repair_split=True
            )
            assert torch.equal(plan["meta"], forced["meta"]) and torch.equal(
                plan["hdr"], forced["hdr"]
            )
            assert not torch.equal(plan["meta"], off["meta"])
    pair = plan_vsa_sm90(
        _random_mask(2, 160, 16, 12, seed=14), sms=132, gqa=4, repair_split=rule
    )
    assert (
        pair["mode"] == "pair"
        and not pair["repair_split"]
        and pair["repair_rule"] == rule
    )
    mha = plan_vsa_sm90(_replicated(multi, 4), sms=132, mode="split", repair_split=rule)
    assert not mha["repair_split"] and mha["num_tiles"] == 8 * 32
    with pytest.raises(ValueError, match="repair_split"):
        plan_vsa_sm90(multi, sms=132, gqa=4, repair_split="always")


@pytest.mark.parametrize("g", [2, 4, 8])
def test_gqa_small_plan_is_the_replicated_masks_rows_plus_g_shift(g):
    """``plan_small(mask_kv, gqa=g)`` is byte for byte ``plan_small(mask_rep)`` (one row per query head and
    query block, the kernel's ``tile = head * MB + qb`` decode) plus the ``g_shift`` scalar; ``small_route``
    applies the MHA thresholds to the replicated mask."""
    for h_kv, mb, nb, capacity, kwargs in (
        (1, 4, 8, 3, {"kmax": 3}),
        (2, 4, 8, 4, {"kmax": 4}),
        (1, 4, 64, 6, {"kmax": 6, "split": True}),
        (1, 4, 16, 8, {"kmax": 4, "cluster": 2}),
        (1, 2, 64, 18, {"kmax": 6, "split": True, "cluster": 3}),
    ):
        mask = _random_mask(h_kv, mb, nb, capacity, seed=mb + nb, ragged=True)
        gqa = plan_small(mask, gqa=g, **kwargs)
        mha = plan_small(_replicated(mask, g), **kwargs)
        assert (
            torch.equal(gqa["plan"], mha["plan"]) and gqa["plan"].dtype == torch.int16
        )
        assert (gqa["gqa"], gqa["g_shift"], gqa["H_kv"], gqa["H"]) == (
            g,
            g.bit_length() - 1,
            h_kv,
            g * h_kv,
        )
        for key in ("num_tiles", "num_items", "kmax", "split", "cluster", "MB", "NB"):
            assert gqa[key] == mha[key], key
        assert (mha["gqa"], mha["g_shift"], mha["H_kv"]) == (1, 0, g * h_kv)
    for h_kv, mb, nb, capacity in (
        (1, 4, 8, 3),
        (2, 16, 16, 4),
        (4, 17, 16, 4),  # 68 x g tiles: over one wave for g >= 2
        (8, 4, 4, 1),
        (2, 32, 32, 8),  # 8 blocks > KMAX 6
        (1, 8, 64, 6),
    ):
        mask = _random_mask(h_kv, mb, nb, capacity, seed=7, ragged=False)
        assert small_route(mask, sms=132, cluster_capacity=CAP, gqa=g) == small_route(
            _replicated(mask, g), sms=132, cluster_capacity=CAP
        )


def test_gqa_small_k4_prefetch_rule(monkeypatch):
    """``small_prefetch``: 2 for the unsplit, non-cluster FP8 grouped-query kmax-4 plan of at least
    ``K4_PREFETCH2_MIN_GRID_GQA`` CTAs while the switch is on (the stage ``small_fp8_gqa_k4p2``), None
    below it, at switch 0, on the BF16 route, on the MHA route (g = 1) and on every other small
    configuration."""
    grids = (1, 32, 64, 127, 128, 132, 256, 1750)
    on = planner.K4_PREFETCH2_MIN_GRID_GQA
    assert [
        planner.small_prefetch(4, False, 0, num_items=n, fp8=True, gqa=4) for n in grids
    ] == [2 if on and n >= on else None for n in grids]
    monkeypatch.setattr(planner, "K4_PREFETCH2_MIN_GRID_GQA", 128)
    for g in (2, 4, 8):
        assert [
            planner.small_prefetch(4, False, 0, num_items=n, fp8=True, gqa=g)
            for n in grids
        ] == [None, None, None, None, 2, 2, 2, 2]
        assert all(
            planner.small_prefetch(4, False, 0, num_items=n, fp8=False, gqa=g) is None
            for n in grids
        )
    assert all(
        planner.small_prefetch(4, False, 0, num_items=n, fp8=True, gqa=1) is None
        for n in grids
    )
    for kmax, split, cluster in (
        (1, False, 0),
        (3, False, 0),
        (6, False, 0),
        (4, True, 0),
        (6, True, 0),
        (4, False, 2),
        (4, False, 4),
        (6, True, 4),
    ):
        assert all(
            planner.small_prefetch(kmax, split, cluster, num_items=n, fp8=True, gqa=4)
            is None
            for n in grids
        ), (kmax, split, cluster)
    monkeypatch.setattr(planner, "K4_PREFETCH2_MIN_GRID_GQA", 0)
    assert all(
        planner.small_prefetch(4, False, 0, num_items=n, fp8=True, gqa=4) is None
        for n in grids
    )
    for g in (0, 3, 5):
        with pytest.raises(ValueError, match=r"g in \(2, 4, 8\)"):
            planner.small_prefetch(4, False, 0, num_items=128, fp8=True, gqa=g)


def test_gqa_group_is_validated_and_g_1_is_the_mha_planner():
    """``gqa`` must be 1 (the MHA plan, bit for bit) or 2 / 4 / 8; every planner entry refuses other values
    with the backend's message."""
    assert GQA_GROUPS == (1, 2, 4, 8)
    for g in GQA_GROUPS:
        assert planner.gqa_group(g) == g
    mask = _random_mask(2, 4, 8, 3, seed=1)
    for g in (0, 3, 5, 6, 16, -4):
        for call in (
            lambda g=g: planner.gqa_group(g),
            lambda g=g: plan_vsa_sm90(mask, sms=132, gqa=g),
            lambda g=g: plan_small(mask, kmax=3, gqa=g),
            lambda g=g: small_route(mask, sms=132, cluster_capacity=CAP, gqa=g),
        ):
            with pytest.raises(ValueError, match=r"g in \(2, 4, 8\)"):
                call()
    one = plan_vsa_sm90(mask, sms=132, gqa=1)
    base = plan_vsa_sm90(mask, sms=132)
    assert torch.equal(one["meta"], base["meta"]) and torch.equal(
        one["hdr"], base["hdr"]
    )
    assert (one["gqa"], one["H_kv"], one["H"]) == (1, 2, 2)
    assert torch.equal(
        plan_small(mask, kmax=3, gqa=1)["plan"], plan_small(mask, kmax=3)["plan"]
    )
    # an empty query block of a kv head is refused like the MHA plan's
    empty = mask.clone()
    empty[1, 2] = False
    with pytest.raises(ValueError, match="empty"):
        plan_vsa_sm90(empty, sms=132, gqa=4)


def _cta_tiles(plan):
    """Per-CTA lists of (head, positions) read back from the plan rows."""
    meta = plan["meta"]
    return [
        [(int(meta[r, 0]), int(meta[r, META_NSEQ])) for r in rows]
        for rows in _tile_rows(plan)
    ]


def _static_tiles(plan):
    """Per-CTA (head, positions) of the static rows only (the pool is run-time)."""
    meta, stride = plan["meta"], plan["tile_stride"]
    return [
        [
            (int(meta[c * stride + i, 0]), int(meta[c * stride + i, META_NSEQ]))
            for i in range(int(meta[c * stride, META_NTILES]))
        ]
        for c in range(plan["num_ctas"])
    ]


def test_plan_ragged_tile_order_balances_when_kv_fits_l2():
    # 7 heads x 64 query blocks over 64 KV blocks: K+V = 14.7 MB, under the
    # L2 budget, and more tiles than SMs.  The global LPT order hands every
    # CTA one of the largest tiles first; no later tile is longer than any
    # CTA's first one.
    mask = _random_mask(7, 64, 64, 16, ragged=True)
    plan = plan_vsa_sm90(mask, mode="split", sms=132)
    ctas = _cta_tiles(plan)
    assert plan["num_ctas"] == 132 and sum(map(len, ctas)) == 7 * 64
    first = min(t[0][1] for t in ctas)
    later = max((p for t in ctas for _, p in t[1:]), default=0)
    assert first >= later
    loads = [sum(p for _, p in t) for t in ctas]
    assert max(loads) <= sum(loads) / len(loads) + max(p for t in ctas for _, p in t)


def test_plan_ragged_tile_order_stays_head_major_above_l2_budget():
    # 8 heads x 512 KV blocks: K+V = 268 MB, above the budget, so each CTA's
    # tiles arrive in head order (the concurrently running CTAs share a head).
    mask = _random_mask(8, 64, 512, 16, ragged=True)
    plan = plan_vsa_sm90(mask, mode="split", sms=132)
    assert not plan["queue"] and plan["pool"] == []  # ragged: static kernel
    for tiles in _static_tiles(plan):
        heads = [h for h, _ in tiles]
        assert heads == sorted(heads)


def test_plan_static_uniform_tiles_in_lpt_order(monkeypatch):
    """A static uniform plan lists its tiles costliest first - over all
    heads when the plan has one round (no more tiles than CTAs: the CTAs launch in index order, so
    the longest tiles start first), within each head otherwise (head-major kept: the concurrently
    streaming CTAs share one head's K/V in L2).  The tile set is unchanged, only the CTA that runs a
    tile moves, so the output bits are those of the legacy order.  Pair tiles of 16-block selections
    have equal modelled cost and a load of 16..32 positions, so the order follows the positions."""
    mask = _topk_mask(2, 64, 64, 16)  # 2 heads x 32 pair tiles: one round on 132 CTAs
    plan = plan_vsa_sm90(mask, mode="pair", sms=132)
    assert not plan["queue"] and plan["num_tiles"] == 64 == plan["num_ctas"]
    tiles = _static_tiles(plan)
    positions = [t[0][1] for t in tiles]
    assert positions == sorted(positions, reverse=True) and positions[0] > positions[-1]
    monkeypatch.setattr(planner, "UNIFORM_LPT", False)
    legacy = _static_tiles(plan_vsa_sm90(mask, mode="pair", sms=132))
    assert sorted(t[0] for t in legacy) == sorted(t[0] for t in tiles)
    assert [t[0][0] for t in legacy] == sorted(
        t[0][0] for t in legacy
    )  # head-major, as planned
    assert legacy != tiles
    monkeypatch.setattr(planner, "UNIFORM_LPT", True)
    # several rounds: 2 heads x 128 pair tiles on 132 CTAs keep the head-major order per CTA
    multi = plan_vsa_sm90(_topk_mask(2, 256, 64, 16), mode="pair", sms=132)
    assert not multi["queue"] and multi["num_tiles"] == 256
    for cta in _static_tiles(multi):
        heads = [h for h, _ in cta]
        assert heads == sorted(heads)


def test_plan_queue_layout_for_long_uniform_plans(monkeypatch):
    """A uniform plan with >= POOL_MIN_TILES tiles per CTA takes the queue kernel: every
    CTA keeps its first (header) tile as its one static row (``tile_stride`` 1), every
    other tile goes to the pool in ascending tile order (head-major, query-block order
    within a head: the planner's tile order of a uniform plan), one all-zero sentinel row
    per CTA follows; below the threshold the layout is the static one."""
    module = planner

    mask = _topk_mask(7, 512, 512, 64)  # 1792 pair tiles = 14 x 128
    plan = plan_vsa_sm90(mask, mode="pair", sms=132)
    lists = plan["tile_lists"]
    g, stride = plan["num_ctas"], plan["tile_stride"]
    assert (
        POOL_ROUNDS >= 14
    )  # the kernel's loop bound admits the whole list of every CTA here
    assert plan["queue"] and g == 128 and stride == 1
    assert len(plan["pool"]) == 1792 - g
    assert plan["num_rows"] == g * stride + len(plan["pool"]) + g
    meta = plan["meta"]
    assert all(int(meta[c * stride, META_NTILES]) == 1 for c in range(g))
    firsts = [lst[0] for lst in lists]
    for c, t in enumerate(firsts):
        assert int(meta[c * stride, 0]) == t // 256  # 256 pair tiles per head
    expected_pool = sorted(t for lst in lists for t in lst[1:])
    assert plan["pool"] == expected_pool and sorted(firsts + expected_pool) == list(
        range(1792)
    )
    pool_rows = [
        tuple(int(v) for v in meta[g * stride + j, :2])
        for j in range(len(plan["pool"]))
    ]
    assert [head for head, _ in pool_rows] == [
        t // 256 for t in expected_pool
    ]  # head-major
    for head in range(
        7
    ):  # query-block order within a head: a pair's first block only increases
        qb0 = [qb for h, qb in pool_rows if h == head]
        assert qb0 == sorted(qb0) and len(set(qb0)) == len(qb0)
    assert all(
        int(meta[g * stride + j, META_NTILES]) == 1 for j in range(len(plan["pool"]))
    )
    assert not meta[g * stride + len(plan["pool"]) :].any()
    assert sum(len(t) for t in _cta_tiles(plan)) == 1792
    monkeypatch.setattr(module, "POOL_MIN_TILES", 15)
    # the static layout below the threshold; UNIFORM_LPT off so the per-CTA lists keep the
    # legacy (head-major, query-block) order the queue plan above was built on
    monkeypatch.setattr(module, "UNIFORM_LPT", False)
    static = plan_vsa_sm90(mask, mode="pair", sms=132)
    assert not static["queue"] and static["tile_stride"] == 14 and static["pool"] == []
    assert static["num_rows"] == 128 * 14 and static["tile_lists"] == lists
    monkeypatch.setattr(module, "UNIFORM_LPT", True)
    # with the LPT order the same tiles are listed, costliest first within each head
    ordered = plan_vsa_sm90(mask, mode="pair", sms=132)
    assert not ordered["queue"] and ordered["tile_stride"] == 14
    assert sorted(t for lst in ordered["tile_lists"] for t in lst) == list(range(1792))
    assert all(int(static["meta"][c * 14, META_NTILES]) == 14 for c in range(128))
    # the kernel's loop bound: a plan longer than POOL_ROUNDS + 1 tiles per CTA keeps the surplus static
    monkeypatch.setattr(module, "POOL_MIN_TILES", 12)
    monkeypatch.setattr(module, "POOL_ROUNDS", 10)
    capped = plan_vsa_sm90(mask, mode="pair", sms=132)
    assert (
        capped["queue"]
        and capped["tile_stride"] == 4
        and len(capped["pool"]) == 128 * 10
    )
    assert all(int(capped["meta"][c * 4, META_NTILES]) == 4 for c in range(128))


def test_plan_queue_threshold_and_tiny_tile_exception(monkeypatch):
    """Uniform plans take the queue from POOL_MIN_TILES tiles per CTA, or from
    POOL_SMALL_MIN_TILES tiles when no tile exceeds POOL_SMALL_TILE_POS ring positions
    (the tiny-tile exception: measured +4.2 % / +4.3 % on h8-m8192-k2, whose split tiles
    hold one position; six-position pair tiles at 7..8 per CTA (h16-m8192-k5) measured
    -1.0 % against the static kernel, so the exception stops at one position);
    8 tiles per CTA of 8 positions keep the static kernel."""
    module = planner

    assert (POOL_MIN_TILES, POOL_SMALL_MIN_TILES, POOL_SMALL_TILE_POS) == (12, 7, 1)
    big = _topk_mask(16, 64, 256, 16)  # split, 1024 tiles = 8 x 128 of 8 positions
    plan = plan_vsa_sm90(big, sms=132)
    assert plan["num_ctas"] == 128 and min(map(len, plan["tile_lists"])) == 8
    assert max(p for t in _cta_tiles(plan) for _, p in t) == 8 and not plan["queue"]
    tiny = _topk_mask(8, 128, 128, 2)  # 1024 tiles of <= 2 positions, 8 per CTA
    tiny_plan = plan_vsa_sm90(tiny, sms=132)
    assert min(map(len, tiny_plan["tile_lists"])) == 8
    tiny_pos = max(p for t in _cta_tiles(tiny_plan) for _, p in t)
    assert tiny_pos <= 2 and tiny_plan["queue"] == (tiny_pos <= POOL_SMALL_TILE_POS)
    # the boundary of the exception: one-position tiles join, two-position tiles do not
    # (split tiles hold half of the query block's KV blocks per warpgroup: 1..2 blocks -> one
    # ring position, 3..4 blocks -> two)
    one = plan_vsa_sm90(_topk_mask(8, 128, 256, 2), mode="split", sms=132)
    two = plan_vsa_sm90(_topk_mask(8, 128, 256, 4), mode="split", sms=132)
    assert min(map(len, one["tile_lists"])) == min(map(len, two["tile_lists"])) == 8
    assert max(p for t in _cta_tiles(one) for _, p in t) == 1 and one["queue"]
    assert max(p for t in _cta_tiles(two) for _, p in t) == 2 and not two["queue"]
    # six-position tiles at 8 per CTA stay static since the exception narrowed
    six = plan_vsa_sm90(_topk_mask(8, 128, 256, 12), mode="split", sms=132)
    assert min(map(len, six["tile_lists"])) == 8
    assert max(p for t in _cta_tiles(six) for _, p in t) == 6 and not six["queue"]
    # fewer than POOL_SMALL_MIN_TILES tiny tiles per CTA stay static
    few = plan_vsa_sm90(_topk_mask(6, 128, 256, 1), mode="split", sms=132)
    assert min(map(len, few["tile_lists"])) == 6 and not few["queue"]
    monkeypatch.setattr(module, "POOL_SMALL_TILE_POS", 0)  # disables the exception
    assert not plan_vsa_sm90(tiny, sms=132)["queue"]
    monkeypatch.setattr(module, "POOL_SMALL_TILE_POS", 8)
    assert plan_vsa_sm90(big, sms=132)["queue"]
    monkeypatch.setattr(module, "POOL_SMALL_MIN_TILES", 9)
    assert not plan_vsa_sm90(big, sms=132)["queue"]


def test_plan_compact_route_rule(monkeypatch):
    """Uniform pair plans of short tiles with many tiles per CTA run the compact static program
    (tiles of at most COMPACT_MAX_TILE_POS ring positions, at least COMPACT_MIN_TILES tiles per
    CTA): h16-m8192-k5 (1024 pair tiles of <= 6 positions, 7..8 per CTA) measured -3 % with the
    lever form of the static kernel and 1.002..1.005 x without it, while 8-position pair tiles at
    4 per CTA (h32-m2048-k8) gain 2.6 % with the levers; split, ragged, queue and short plans keep
    their programs."""
    module = planner

    assert (COMPACT_MAX_TILE_POS, COMPACT_MIN_TILES) == (6, 4)
    k5_mask = _topk_mask(16, 128, 128, 5)  # h16-m8192-n8192-k5
    k5 = plan_vsa_sm90(k5_mask, sms=132)
    assert k5["mode"] == "pair" and not k5["queue"] and k5["compact"]
    assert min(map(len, k5["tile_lists"])) >= COMPACT_MIN_TILES
    assert max(p for t in _cta_tiles(k5) for _, p in t) <= COMPACT_MAX_TILE_POS
    k8 = plan_vsa_sm90(
        _topk_mask(32, 32, 32, 8), sms=132
    )  # h32-m2048-n2048-k8: 8-position pair tiles
    assert k8["mode"] == "pair" and not k8["queue"] and not k8["compact"]
    assert max(p for t in _cta_tiles(k8) for _, p in t) > COMPACT_MAX_TILE_POS
    assert not plan_vsa_sm90(k5_mask, mode="split", sms=132)["compact"]
    ragged = k5_mask.clone()
    ragged[:, ::2] &= ~_topk_mask(16, 64, 128, 1).repeat_interleave(1, dim=1)[
        :, : ragged[:, ::2].shape[1]
    ]
    assert (
        int(ragged.sum(-1).min()) < 5
        and not plan_vsa_sm90(ragged, mode="pair", sms=132)["compact"]
    )
    short = plan_vsa_sm90(
        _topk_mask(16, 32, 128, 5), mode="pair", sms=132
    )  # 256 pair tiles on 128 CTAs: fewer than 4 per CTA (mode=None picks split)
    assert short["mode"] == "pair" and not short["compact"]
    assert min(map(len, short["tile_lists"])) < COMPACT_MIN_TILES
    long = plan_vsa_sm90(
        _topk_mask(16, 256, 128, 5), sms=132
    )  # >= POOL_MIN_TILES per CTA: queue, not compact
    assert long["queue"] and not long["compact"]
    monkeypatch.setattr(module, "COMPACT_MIN_TILES", 9)
    assert not plan_vsa_sm90(k5_mask, sms=132)["compact"]
    monkeypatch.setattr(module, "COMPACT_MIN_TILES", 4)
    monkeypatch.setattr(module, "COMPACT_MAX_TILE_POS", 4)
    assert not plan_vsa_sm90(k5_mask, sms=132)["compact"]


def test_plan_pin_route_rule(monkeypatch):
    """Static grouped-query plans whose lightest CTA runs at least PIN_MIN_CTA_LOOP_ITERS main-loop
    iterations (``cta_loop_iters``: per tile the larger of the two warpgroups' owned ring positions
    minus one, summed over the CTA's tiles) carry ``pin`` and run the position-selected FP8 static
    program (``attention_fp8_gqa_pin``: h8-m4096-n4096-k32 at g = 4 is 256 head-pair tiles of 16
    positions on 128 CTAs, 30 iterations per warpgroup); shorter static plans (h8-m2048-n2048-k16 at
    g = 4: 7 iterations), queue, compact and tagged plans and every MHA plan carry ``pin`` False and
    keep their programs.  The planner decides the key inside the plan and ``fp8_gqa_pin`` reads it;
    the kernel-tests reference planner states the same statistic."""
    module = planner

    assert planner.PIN_MIN_CTA_LOOP_ITERS == 8
    g = 4
    long_mask = _topk_mask(2, 64, 64, 32)  # h8-m4096-n4096-k32 at g = 4
    long = plan_vsa_sm90(long_mask, sms=132, gqa=g)
    assert long["mode"] == "pair" and not long["queue"] and not long["compact"]
    assert not long["tagged"] and long["gqa"] == g
    assert long["cta_loop_iters"] >= planner.PIN_MIN_CTA_LOOP_ITERS
    assert long["pin"] is True
    assert planner.fp8_gqa_pin(long) is True and planner.fp8_gqa_xshare(long) is False
    # head-pair tiles: both warpgroups own every position, so the statistic is the lightest CTA's
    # sum over its tiles of (positions - 1)
    lightest = min(sum(p - 1 for _, p in tiles) for tiles in _cta_tiles(long))
    assert lightest == long["cta_loop_iters"]
    # h8-m2048-n2048-k16 at g = 4: 8-position tiles
    short_mask = _topk_mask(2, 32, 32, 16)
    short = plan_vsa_sm90(short_mask, sms=132, gqa=g)
    assert not short["queue"]
    assert short["cta_loop_iters"] < planner.PIN_MIN_CTA_LOOP_ITERS
    assert short["pin"] is False and planner.fp8_gqa_pin(short) is False
    # h16-m8192-n8192-k5: compact
    compact = plan_vsa_sm90(_topk_mask(4, 128, 128, 5), sms=132, gqa=g)
    assert compact["compact"] and compact["pin"] is False
    # 2112 head-pair tiles: queue
    queue = plan_vsa_sm90(_topk_mask(2, 528, 64, 8), sms=132, gqa=g)
    assert queue["queue"] and queue["pin"] is False
    assert planner.fp8_gqa_pin(queue) is False
    mha = plan_vsa_sm90(_replicated(long_mask, g), sms=132)
    assert mha["gqa"] == 1 and mha["pin"] is False and planner.fp8_gqa_pin(mha) is False
    # the key is read, never recomputed: no key or an MHA plan is an unrouted plan
    assert planner.fp8_gqa_pin({"gqa": 4}) is False
    assert planner.fp8_gqa_pin({"gqa": 1, "pin": True}) is False
    # the threshold is the route; the plan bytes do not depend on it
    monkeypatch.setattr(module, "PIN_MIN_CTA_LOOP_ITERS", 10**9)
    off = plan_vsa_sm90(long_mask, sms=132, gqa=g)
    assert off["pin"] is False and off["cta_loop_iters"] == long["cta_loop_iters"]
    assert torch.equal(off["meta"], long["meta"])
    assert torch.equal(off["hdr"], long["hdr"])
    monkeypatch.setattr(module, "PIN_MIN_CTA_LOOP_ITERS", short["cta_loop_iters"])
    assert plan_vsa_sm90(short_mask, sms=132, gqa=g)["pin"] is True
    assert plan_vsa_sm90(_replicated(short_mask, g), sms=132)["pin"] is False


def test_plan_ragged_queue_threshold_and_pool_order():
    """Ragged plans join the queue from POOL_RAGGED_MIN_TILES tiles per CTA (measured +0.7..+1.7 %
    at >= 29 tiles per CTA, -1.3..-3.3 % at 12..25); their pool keeps the planner's tile order
    above the L2 budget: head-major, cost-descending within a head."""
    assert POOL_RAGGED_MIN_TILES == 28
    long = _random_mask(
        8, 1024, 128, 16, seed=5, ragged=True
    )  # 8192 tiles; K+V = 33.5 MB > budget
    plan = plan_vsa_sm90(long, mode="split", sms=132)
    g, stride, lists = plan["num_ctas"], plan["tile_stride"], plan["tile_lists"]
    assert (
        min(map(len, lists)) >= POOL_RAGGED_MIN_TILES and plan["queue"] and stride == 1
    )
    assert plan["pool"] == sorted(t for lst in lists for t in lst[1:])
    meta = plan["meta"]
    assert all(int(meta[c, META_NTILES]) == 1 for c in range(g))
    pool_rows = [meta[g * stride + j] for j in range(len(plan["pool"]))]
    heads = [int(r[0]) for r in pool_rows]
    assert heads == sorted(heads)
    for head in range(8):
        costs = [-(-int(r[META_NSEQ]) // 2) for r in pool_rows if int(r[0]) == head]
        assert costs == sorted(costs, reverse=True), head
    assert not meta[g * stride + len(plan["pool"]) :].any()
    assert sum(len(t) for t in _cta_tiles(plan)) == 8192
    # Long-head rule: ragged plans with >= POOL_RAGGED_LONG_HEAD_MIN_TILES tiles per CTA whose
    # heads hold POOL_RAGGED_LONG_HEAD_WINDOWS grids' worth of tiles join the queue below the
    # 28-tile threshold (h3-m109632-k64 ragged: +1.4..1.5 %); the same tiles spread over many
    # short heads (h12-m32768: -2.8 %) and plans below the per-CTA floor stay static.
    assert (POOL_RAGGED_LONG_HEAD_MIN_TILES, POOL_RAGGED_LONG_HEAD_WINDOWS) == (18, 4)
    long_heads = _random_mask(
        3, 1024, 128, 16, seed=5, ragged=True
    )  # 3072 tiles: 18..28 per CTA, 1024 per head >= 4 x 132
    joined = plan_vsa_sm90(long_heads, mode="split", sms=132)
    assert (
        POOL_RAGGED_LONG_HEAD_MIN_TILES
        <= min(map(len, joined["tile_lists"]))
        < POOL_RAGGED_MIN_TILES
    )
    assert joined["queue"] and joined["tile_stride"] == 1
    many_heads = _random_mask(
        12, 256, 128, 16, seed=5, ragged=True
    )  # 3072 tiles over 12 heads: 256 per head < 4 x 132
    static = plan_vsa_sm90(many_heads, mode="split", sms=132)
    assert min(map(len, static["tile_lists"])) < POOL_RAGGED_MIN_TILES
    assert (
        not static["queue"]
        and static["pool"] == []
        and static["num_rows"] == static["num_ctas"] * static["tile_stride"]
    )
    few = plan_vsa_sm90(
        _random_mask(3, 512, 128, 16, seed=5, ragged=True), mode="split", sms=132
    )
    assert (
        min(map(len, few["tile_lists"])) < POOL_RAGGED_LONG_HEAD_MIN_TILES
        and not few["queue"]
    )


def test_plan_mode_selection_prefers_split_below_full_occupancy():
    mask = _random_mask(8, 16, 16, 12, ragged=False)
    assert plan_vsa_sm90(mask, sms=132)["mode"] == "split"
    dense = _random_mask(8, 64, 64, 32, ragged=False)
    assert plan_vsa_sm90(dense, sms=132)["mode"] == "pair"


requires_hopper = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="Requires an SM90 GPU",
)


def _inputs(h, mb, nb, seed=7):
    """Deterministic BF16 Q/K/V (HND) for the kernel-vs-reference checks."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    return tuple(
        torch.randn(
            (h, length * 64, 128), device="cuda", dtype=torch.bfloat16, generator=g
        )
        for length in (mb, nb, nb)
    )


def _reference(q, k, v, mask, scale):
    """Independent FP32 reference: masked dense softmax attention per head.

    Returned unrounded (FP32): the BF16 kernel output is compared against the
    exact value, so the check bounds the kernel's own rounding (at most half a
    BF16 ulp plus the FP32 accumulation difference) instead of asking two
    independently rounded BF16 results to agree, which any two implementations
    fail by one ulp (0.03125 for |o| in [4, 8)) on a fraction of the elements.
    """
    scores = torch.einsum("hmd,hnd->hmn", q.float(), k.float()) * float(scale)
    dense = mask.to(q.device).repeat_interleave(64, dim=1).repeat_interleave(64, dim=2)
    scores.masked_fill_(~dense, float("-inf"))
    return torch.einsum("hmn,hnd->hmd", torch.softmax(scores, dim=-1), v.float())


@requires_hopper
@pytest.mark.parametrize(
    "h,mb,nb,capacity,scale",
    [
        (1, 1, 1, 1, None),
        (4, 4, 4, 1, 0.5),
        (8, 4, 4, 3, None),
        (8, 16, 16, 4, -0.125),
        (2, 8, 32, 6, 0.0),
    ],
)
def test_small_and_persistent_routes_agree(h, mb, nb, capacity, scale):
    """Both kernels implement the same contract: force each route on a small problem."""
    mask = _random_mask(h, mb, nb, capacity, seed=11, ragged=True, device="cuda")
    rows = torch.full((h, mb), 64, dtype=torch.int32, device="cuda")
    cols = torch.full((h, nb), 64, dtype=torch.int32, device="cuda")
    q, k, v = _inputs(h, mb, nb)
    outputs = {}
    for route in ("small", "persistent"):
        plan = CakeVsaSm90Plan(
            "cuda", mask, rows, cols, h, h, 128, sm_scale=scale, route=route
        )
        assert plan.mode == ("small" if route == "small" else plan.mode)
        outputs[route] = plan.run(q, k, v).float()
    assert plan.mode in ("pair", "split")
    reference = _reference(q, k, v, mask, 128**-0.5 if scale is None else scale).float()
    for route, out in outputs.items():
        torch.testing.assert_close(out, reference, atol=1e-2, rtol=1e-2)
        assert float((out - reference).abs().max()) <= 0.03, route
    # Auto routing picks the small kernel for every one of these problems.
    auto = CakeVsaSm90Plan("cuda", mask, rows, cols, h, h, 128, sm_scale=scale)
    assert auto.mode == "small"


@requires_hopper
@pytest.mark.parametrize(
    "h,mb,nb,capacity,scale,ragged",
    [
        (1, 16, 16, 16, None, False),  # the benchmark's h1-m1024-k16 row
        (1, 16, 64, 16, 0.5, True),
        (2, 8, 64, 64, None, True),  # 64 blocks -> up to 11 slices at KMAX 6
        (1, 1, 8, 6, 1e-7, False),  # one query block, two slices of KMAX 4
        (3, 5, 10, 9, 0.0, True),
        (3, 5, 10, 9, -0.125, True),
    ],
)
def test_split_route_against_persistent(h, mb, nb, capacity, scale, ragged):
    """The split-KV small kernel matches the persistent kernel and the reference; runs twice."""
    mask = _random_mask(h, mb, nb, capacity, seed=13, ragged=ragged, device="cuda")
    rows = torch.full((h, mb), 64, dtype=torch.int32, device="cuda")
    cols = torch.full((h, nb), 64, dtype=torch.int32, device="cuda")
    q, k, v = _inputs(h, mb, nb)
    split = CakeVsaSm90Plan(
        "cuda", mask, rows, cols, h, h, 128, sm_scale=scale, route="smallsplit"
    )
    assert split.mode == "smallsplit" and split.small_split
    assert split.num_items > split.num_tiles
    first = split.run(q, k, v).float()
    second = split.run(q, k, v).float()
    torch.cuda.synchronize()
    # The last-arriving CTA resets its query block's counter: the plan replays.
    assert int(split.counters.abs().sum()) == 0
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    persistent = CakeVsaSm90Plan(
        "cuda", mask, rows, cols, h, h, 128, sm_scale=scale, route="persistent"
    )
    reference = _reference(q, k, v, mask, 128**-0.5 if scale is None else scale).float()
    for name, out in (
        ("smallsplit", first),
        ("persistent", persistent.run(q, k, v).float()),
    ):
        torch.testing.assert_close(out, reference, atol=1e-2, rtol=1e-2)
        assert float((out - reference).abs().max()) <= 0.03, name
    # The automatic route follows ``small_route`` (unsplit variant first, then
    # split-KV when the grid cannot fill the persistent kernel).
    auto = CakeVsaSm90Plan("cuda", mask, rows, cols, h, h, 128, sm_scale=scale)
    rule = small_route(
        mask,
        sms=torch.cuda.get_device_properties(0).multi_processor_count,
        cluster_capacity=cluster_capacity(torch.cuda.current_device()),
    )
    expected_mode = _mode_of(rule)
    assert auto.mode == expected_mode


def _mode_of(rule):
    if rule is None:
        return "persistent"
    if rule[2]:
        return "smallsplitcluster" if rule[1] else "smallcluster"
    return "smallsplit" if rule[1] else "small"


@requires_hopper
@pytest.mark.parametrize(
    "h,mb,nb,capacity,scale,ragged",
    [
        (1, 16, 16, 16, None, False),  # the benchmark's h1-m1024-k16 row -> k4c4
        (1, 16, 64, 16, 0.5, True),  # padded slices
        (
            4,
            16,
            16,
            8,
            None,
            True,
        ),  # the ragged 8-block row (auto routes it to the 4 x 2 cluster)
        (3, 5, 10, 9, -0.125, True),
        (1, 1, 2, 2, 1e-7, False),  # two blocks over four ranks
        (2, 8, 64, 18, 0.0, False),  # 18 blocks: kmax 6 x 3
    ],
)
def test_cluster_route_against_persistent(h, mb, nb, capacity, scale, ragged):
    """The cluster / DSM-merge small kernel matches the persistent kernel and the reference; runs twice, bit-exact."""
    mask = _random_mask(h, mb, nb, capacity, seed=19, ragged=ragged, device="cuda")
    rows = torch.full((h, mb), 64, dtype=torch.int32, device="cuda")
    cols = torch.full((h, nb), 64, dtype=torch.int32, device="cuda")
    q, k, v = _inputs(h, mb, nb)
    plan = CakeVsaSm90Plan(
        "cuda", mask, rows, cols, h, h, 128, sm_scale=scale, route="smallcluster"
    )
    assert plan.mode == "smallcluster" and plan.small_cluster > 1
    assert plan.num_items == plan.num_tiles * plan.small_cluster
    first = plan.run(q, k, v).float()
    second = plan.run(q, k, v).float()
    torch.cuda.synchronize()
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    persistent = CakeVsaSm90Plan(
        "cuda", mask, rows, cols, h, h, 128, sm_scale=scale, route="persistent"
    )
    reference = _reference(q, k, v, mask, 128**-0.5 if scale is None else scale).float()
    for name, out in (
        ("smallcluster", first),
        ("persistent", persistent.run(q, k, v).float()),
    ):
        torch.testing.assert_close(out, reference, atol=1e-2, rtol=1e-2)
        assert float((out - reference).abs().max()) <= 0.03, name
    auto = CakeVsaSm90Plan("cuda", mask, rows, cols, h, h, 128, sm_scale=scale)
    rule = small_route(
        mask,
        sms=torch.cuda.get_device_properties(0).multi_processor_count,
        cluster_capacity=cluster_capacity(torch.cuda.current_device()),
    )
    assert auto.mode == _mode_of(rule)


@requires_hopper
@pytest.mark.parametrize(
    "h,mb,nb,capacity,scale,ragged,stage",
    [
        (
            1,
            16,
            64,
            18,
            None,
            False,
            "small_k6sc3",
        ),  # 16 tiles x 18 blocks: three slices of six
        (
            1,
            16,
            64,
            24,
            None,
            False,
            "small_k6sc4",
        ),  # the h1-m1024-n4096-k24 row (auto routes it here)
        (
            1,
            16,
            64,
            24,
            0.5,
            True,
            "small_k6sc4",
        ),  # padded slices, one-slice tiles leave ranks idle
        (1, 8, 128, 48, None, False, "small_k6sc8"),  # the h1-m512-n8192-k48 row
        (
            4,
            8,
            64,
            30,
            -0.125,
            True,
            "small_k6sc8",
        ),  # 32 tiles x up to 30 blocks: five of eight ranks
        (
            2,
            8,
            64,
            20,
            0.0,
            True,
            "small_k6sc4",
        ),  # scale 0: a uniform softmax over the selection
        (
            1,
            1,
            64,
            7,
            1e-7,
            False,
            "small_k6sc3",
        ),  # one query block, two slices, one idle rank
    ],
)
def test_split_cluster_route_against_persistent(
    h, mb, nb, capacity, scale, ragged, stage
):
    """The split-cluster small kernel (slices of a query block in one thread-block cluster, merged
    column-set by column-set from the workspace) matches the persistent kernel and the reference;
    runs twice bit-exact and leaves the arrival counters it does not use at zero."""
    mask = _random_mask(h, mb, nb, capacity, seed=29, ragged=ragged, device="cuda")
    rows = torch.full((h, mb), 64, dtype=torch.int32, device="cuda")
    cols = torch.full((h, nb), 64, dtype=torch.int32, device="cuda")
    q, k, v = _inputs(h, mb, nb)
    plan = CakeVsaSm90Plan(
        "cuda", mask, rows, cols, h, h, 128, sm_scale=scale, route="smallsplitcluster"
    )
    assert plan.mode == "smallsplitcluster" and plan.small_split and plan.stage == stage
    assert (plan.small_kmax, plan.small_cluster) in SPLIT_CLUSTER_VARIANTS
    assert plan.num_items == plan.num_tiles * plan.small_cluster == plan.num_ctas
    assert plan.partial_o.numel() == plan.num_items * SMALL_ITEM_ELEMS
    first = plan.run(q, k, v).float()
    second = plan.run(q, k, v).float()
    torch.cuda.synchronize()
    assert int(plan.counters.abs().sum()) == 0
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    persistent = CakeVsaSm90Plan(
        "cuda", mask, rows, cols, h, h, 128, sm_scale=scale, route="persistent"
    )
    reference = _reference(q, k, v, mask, 128**-0.5 if scale is None else scale).float()
    for name, out in (
        ("smallsplitcluster", first),
        ("persistent", persistent.run(q, k, v).float()),
    ):
        torch.testing.assert_close(out, reference, atol=1e-2, rtol=1e-2)
        assert float((out - reference).abs().max()) <= 0.03, name
    auto = CakeVsaSm90Plan("cuda", mask, rows, cols, h, h, 128, sm_scale=scale)
    rule = small_route(
        mask,
        sms=torch.cuda.get_device_properties(0).multi_processor_count,
        cluster_capacity=cluster_capacity(torch.cuda.current_device()),
    )
    assert auto.mode == _mode_of(rule)
    if auto.mode == "smallsplitcluster":
        torch.testing.assert_close(auto.run(q, k, v).float(), first, atol=0, rtol=0)


@requires_hopper
def test_split_cluster_plan_stream_and_graph_lifetime():
    """A split-cluster plan owns the split workspace: built on another stream, captured and replayed
    bit-exactly, counters zero after every launch."""
    h, mb, nb = 1, 16, 64
    mask = _random_mask(h, mb, nb, 24, seed=17, ragged=False, device="cuda")
    rows = torch.full((h, mb), 64, dtype=torch.int32, device="cuda")
    cols = torch.full((h, nb), 64, dtype=torch.int32, device="cuda")
    producer = torch.cuda.Stream()
    with torch.cuda.stream(producer):
        plan = CakeVsaSm90Plan(
            "cuda", mask, rows, cols, h, h, 128, route="smallsplitcluster"
        )
    assert plan.mode == "smallsplitcluster" and plan.stage == "small_k6sc4"
    q, k, v = _inputs(h, mb, nb)
    expected = plan.run(q, k, v).clone()
    out = torch.empty((h * mb * 64, 1, 128), device="cuda", dtype=q.dtype)
    plan.run(q, k, v, out=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan.run(q, k, v, out=out)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out.view_as(q), expected, atol=0, rtol=0)
    assert int(plan.counters.abs().sum()) == 0
    q.normal_()
    graph.replay()
    torch.testing.assert_close(
        out.view_as(q).float(),
        _reference(q, k, v, mask, 128**-0.5),
        atol=0.01,
        rtol=0.01,
    )


@requires_hopper
def test_cluster_plan_stream_and_graph_lifetime():
    """A cluster plan owns no device workspace: built on another stream, captured and replayed bit-exactly."""
    h, mb, nb = 1, 16, 16
    mask = _random_mask(h, mb, nb, 16, seed=17, ragged=False, device="cuda")
    rows = torch.full((h, mb), 64, dtype=torch.int32, device="cuda")
    cols = torch.full((h, nb), 64, dtype=torch.int32, device="cuda")
    producer = torch.cuda.Stream()
    with torch.cuda.stream(producer):
        plan = CakeVsaSm90Plan("cuda", mask, rows, cols, h, h, 128)
    assert plan.mode == "smallcluster" and (plan.small_kmax, plan.small_cluster) == (
        4,
        4,
    )
    q, k, v = _inputs(h, mb, nb)
    expected = plan.run(q, k, v).clone()
    out = torch.empty((h * mb * 64, 1, 128), device="cuda", dtype=q.dtype)
    plan.run(q, k, v, out=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan.run(q, k, v, out=out)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out.view_as(q), expected, atol=0, rtol=0)
    q.normal_()
    graph.replay()
    torch.testing.assert_close(
        out.view_as(q).float(),
        _reference(q, k, v, mask, 128**-0.5),
        atol=0.01,
        rtol=0.01,
    )


@requires_hopper
def test_split_plan_stream_and_graph_lifetime():
    """Split workspace built on another stream; graph replays reuse the reset counters."""
    h, mb, nb = 1, 16, 16
    mask = _random_mask(h, mb, nb, 16, seed=17, ragged=False, device="cuda")
    rows = torch.full((h, mb), 64, dtype=torch.int32, device="cuda")
    cols = torch.full((h, nb), 64, dtype=torch.int32, device="cuda")
    producer = torch.cuda.Stream()
    with torch.cuda.stream(producer):
        plan = CakeVsaSm90Plan("cuda", mask, rows, cols, h, h, 128, route="smallsplit")
    assert plan.mode == "smallsplit"
    q, k, v = _inputs(h, mb, nb)
    expected = plan.run(q, k, v).clone()
    out = torch.empty((h * mb * 64, 1, 128), device="cuda", dtype=q.dtype)
    plan.run(q, k, v, out=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan.run(q, k, v, out=out)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out.view_as(q), expected, atol=0, rtol=0)
    assert int(plan.counters.abs().sum()) == 0
    q.normal_()
    graph.replay()
    torch.testing.assert_close(
        out.view_as(q).float(),
        _reference(q, k, v, mask, 128**-0.5),
        atol=0.01,
        rtol=0.01,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="cake (SM90) VSA requires Hopper compute capability 9.0",
)
def test_queue_kernel_matches_the_static_kernel_bit_exact(monkeypatch):
    """The queue kernel only moves tiles between CTAs: its output is bit-identical
    to the static kernel's on the same plan, its queue words return to zero after
    every launch, and a captured graph replays it."""
    module = planner

    h, mb, nb = (
        8,
        528,
        64,
    )  # 2112 pair tiles: 15-16 per CTA on 132 SMs (>= POOL_MIN_TILES)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    mask = _topk_mask(h, mb, nb, 8).cuda()
    rows = torch.full((h, mb), 64, dtype=torch.int32, device="cuda")
    cols = torch.full((h, nb), 64, dtype=torch.int32, device="cuda")
    q, k, v = _inputs(h, mb, nb)
    queued = CakeVsaSm90Plan("cuda", mask, rows, cols, h, h, 128, route="persistent")
    if not queued.queue:
        pytest.skip(f"{sms} SMs: fewer than {POOL_MIN_TILES} tiles per CTA")
    out = torch.empty((h * mb * 64, 1, 128), device="cuda", dtype=q.dtype)
    first = queued.run(q, k, v, out=out).clone()
    torch.cuda.synchronize()
    assert int(queued.dbg[62:64].abs().sum()) == 0
    monkeypatch.setattr(module, "POOL_MIN_TILES", 10**9)
    static = CakeVsaSm90Plan("cuda", mask, rows, cols, h, h, 128, route="persistent")
    assert not static.queue and static.num_ctas == queued.num_ctas
    torch.testing.assert_close(static.run(q, k, v), first, atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        queued.run(q, k, v, out=out)
    for _ in range(3):
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out.view_as(q), first, atol=0, rtol=0)
        assert int(queued.dbg[62:64].abs().sum()) == 0
    reference = _reference(q, k, v, mask, 128**-0.5).float()
    torch.testing.assert_close(first.float(), reference, atol=1e-2, rtol=1e-2)


def _topk_mask(h, mb, nb, k, seed=42):
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.rand((h, mb, nb), generator=g).topk(k, dim=-1).indices
    mask = torch.zeros((h, mb, nb), dtype=torch.bool)
    mask.scatter_(-1, idx, True)
    return mask


def test_plan_prefers_pair_over_short_split_tiles():
    # h3, 75648 tokens, K = 32: 16-position tiles.  The split plan has twice
    # the tiles, each paying the fixed per-tile load and a partial merge; the
    # pair plan measured 6-12 % faster on H100, so the load model must pick it.
    mask = _topk_mask(3, 1182, 1182, 32)
    plan = plan_vsa_sm90(mask, sms=132)
    assert int(plan["hdr"][3]) & 0xF != 1, "pair plan expected"
    assert plan["num_tiles"] == 3 * 1182 // 2


def test_plan_smaller_grid_only_without_an_extra_tile():
    # h7, 109632 tokens, K = 64: 5999 pair tiles, 46 per CTA on the full grid.
    # A smaller grid may take the per-SM rate credit only when its fullest CTA
    # carries the same number of tiles (128 CTAs x 47-48 measured slower).
    mask = _topk_mask(7, 1713, 1713, 64)
    plan = plan_vsa_sm90(mask, mode="pair", sms=132)
    per_cta = [len(t) for t in _cta_tiles(plan)]
    assert max(per_cta) == 46 and plan["num_ctas"] >= 131


def test_plan_uniform_128_cta_grid():
    # 7 x 256 pair tiles = 1792 = 14 x 128: the knee model takes 128 CTAs with
    # the same maximum tile count instead of 132 (measured 1-2 % faster).
    mask = _topk_mask(7, 512, 512, 64)
    plan = plan_vsa_sm90(mask, mode="pair", sms=132)
    per_cta = [len(t) for t in _cta_tiles(plan)]
    assert plan["num_ctas"] == 128 and max(per_cta) == 14


def test_plan_refined_load_model_only_for_long_tiles():
    # h8, 2048 tokens, K = 8: 4-position tiles.  The refined load model would
    # pick the pair plan, which measured 8-14 % slower than the split plan on
    # H100; below REFINED_LOAD_MIN_K blocks per query block the previous
    # model decides (split, two tiles per CTA).
    mask = _topk_mask(8, 32, 32, 8)
    plan = plan_vsa_sm90(mask, sms=132)
    assert int(plan["hdr"][3]) & 0xF == 1, "split plan expected on a short-tile row"
    assert plan["num_tiles"] == 8 * 32
    assert max(len(t) for t in _cta_tiles(plan)) == 2


def test_plan_no_credit_for_intermediate_grids():
    # h7, 109632 tokens, K = 64: 5999 pair tiles.  A 131-CTA grid carries the same
    # 46-tile maximum but measured slower on H100; grids between 128 and the
    # full machine take no rate credit, so the full grid is kept.
    mask = _topk_mask(7, 1713, 1713, 64)
    plan = plan_vsa_sm90(mask, mode="pair", sms=132)
    assert len(_cta_tiles(plan)) == 132


def test_plan_ragged_keeps_the_previous_load_model():
    # h10, 32768 tokens, K = 64 ragged: the refined model would pick 128 CTAs;
    # ragged plans keep the previous model (full grid, raw-position balance).
    g = torch.Generator(device="cpu").manual_seed(42)
    idx = torch.rand((10, 512, 512), generator=g).topk(64, dim=-1).indices
    counts = torch.randint(1, 65, (10, 512), generator=g)
    counts[:, 0] = 64
    mask = torch.zeros((10, 512, 512), dtype=torch.bool)
    mask.scatter_(-1, idx, torch.arange(64) < counts.unsqueeze(-1))
    plan = plan_vsa_sm90(mask, sms=132)
    assert len(_cta_tiles(plan)) == 132


def _odd_single_block_mask():
    """Pair plan with one odd block per head that has a single KV block.

    129 query blocks per head at 64 KV blocks each pair up to 64 pair tiles and
    leave query block 128 (one KV block) alone on warpgroup 0 (tile mode 2).
    """
    g = torch.Generator().manual_seed(3)
    mask = torch.zeros((2, 129, 1024), dtype=torch.bool)
    for head in range(2):
        for row in range(129):
            count = 1 if row == 128 else 64
            mask[head, row, torch.randperm(1024, generator=g)[:count]] = True
    return mask


def test_plan_odd_block_of_a_pair_plan_runs_alone_on_warpgroup_0():
    plan = plan_vsa_sm90(_odd_single_block_mask(), sms=132)
    assert plan["mode"] == "pair"
    meta, stride = plan["meta"], plan["tile_stride"]
    odd = [
        meta[c * stride + i]
        for c in range(plan["num_ctas"])
        for i in range(int(meta[c * stride, META_NTILES]))
        if int(meta[c * stride + i, 3]) == 2
    ]
    assert len(odd) == 2
    for row in odd:
        assert int(row[1]) == int(row[2]) == 128
        assert int(row[META_NSEQ]) == 1
        assert (int(row[META_NOWN]), int(row[META_NOWN + 1])) == (1, 0)


# --------------------------------------------------------------------------- Sage FP8 (E4M3) route
#
# The FP8 stages consume E4M3 q/k/v with per-token / per-16-token / per-channel FP32 scales bound at plan
# time on exactly the BF16 planner's bytes.  Two references: (b) the exact FP32 softmax of
# the dequantized inputs (``_fp8_reference_exact``, atol = rtol = 5e-2) and (a) a replay of the kernel's
# own precision boundary on the plan's schedule (``_fp8_reference_replay``, atol = rtol = 1e-2): per
# warpgroup chain the lazy-max rule (a running maximum that only grows by more than ``FP8_MAX_GROWTH_LOG2``
# triggers a rescale), ``P = exp2(fma(s, qk_sc, FP8_P_QUANT_LOG2 - m))`` formed as the kernels form it (one
# rounding of the argument, the device ``ex2.approx`` exponent; ``_fp8_sage_p``) rounded to E4M3 before the
# P.V product, FP32 row sums of the unrounded P, chains folded with flash-decoding weights in the kernel's
# fixed order,
# ``acc * v_scale / l`` last, rounded to BF16.  The scores of every step and the P.V product of every
# position are accumulated through the sm_90a FP8 WGMMA's own arithmetic (``_fp8_wgmma_scores`` /
# ``_fp8_wgmma_pv``: 13 fractional bits below the step's largest term, truncation toward zero, a
# 14-significant-bit step result; the P.V chain carries the FP32 O accumulator through the alignment set).
# The epilogue (the kernel's ``rcp.approx`` normalisation, the ``v_scale`` multiplies and the BF16 pack) is
# not modelled: the kernel and this replay differ by one BF16 ulp on about 1e-4 of the elements, far inside
# the 1e-2 gate.  The two constants are the kernels' ``P_QUANT_LOG2`` / ``MAX_GROWTH_LOG2`` (the Sage FP8
# values the SM100 kernels use as well).

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
K_GROUP = 16
FP8_P_QUANT_LOG2 = 8.0
FP8_MAX_GROWTH_LOG2 = 0.8073549220576041  # log2(448 / 256)
LOG2E = 1.4426950408889634
E4M3_UNIT_LOG2 = 9  # every E4M3 value is an integer multiple of 2^-9
E4M3_MIN_NORMAL_EXP = -6
WGMMA_K = 32
WGMMA_FRACTION_BITS = 13
WGMMA_RESULT_BITS = 14
# The score accumulate of reference (a): the FP8 WGMMA model (the default) or a float32 matmul (the opt-out).
SCORE_ACCUMULATE_MODES = ("fp8_wgmma", "fp32")
# The P.V accumulate of reference (a): the same model on the carried O accumulator (the default) or
# ``acc + P.V`` in float32 (the opt-out).
PV_ACCUMULATE_MODES = ("fp8_wgmma", "fp32")
# The scale variants: one key per head, in the middle of the sequence, two octaves above the rest (the growth
# branch mid-chain while the other keys' P stay representable), and per-channel V magnitudes over three
# decades below unit scale.
SPIKE_LOG2 = 2.0
V_DYNAMIC_LOG10 = (-3.0, 0.0)


def _e4m3_exponent(x):
    """Unbiased exponent of the E4M3 encoding of the exact E4M3 values ``x`` (float32): ``floor(log2 |x|)`` for
    a normal, -6 for a subnormal, a very negative sentinel for zero."""
    _, e = torch.frexp(x.abs())
    e = e.to(torch.int32) - 1
    e = torch.where(
        x.abs() < 2.0**E4M3_MIN_NORMAL_EXP, torch.full_like(e, E4M3_MIN_NORMAL_EXP), e
    )
    return torch.where(x == 0, torch.full_like(e, -10_000), e)


def _truncate(x, g):
    """``x`` truncated toward zero to a multiple of the power of two ``g``; exact in float32."""
    return torch.trunc(x / g) * g


def _granularity(e, bits, floor_log2=-2 * E4M3_UNIT_LOG2):
    """``2^(e - bits)`` as float32, floored at ``2^floor_log2`` (the product unit ``2^-18`` for the scores, the
    smallest normal exponent for the P.V chain, whose carried accumulator is an arbitrary float32)."""
    ones = torch.ones(e.shape, dtype=torch.float32, device=e.device)
    return torch.ldexp(ones, (e - bits).clamp_min(floor_log2))


def _fp8_wgmma_accumulate(acc, a, b, *, floor_log2):
    """``acc (+) a b^T`` (``acc`` ``[M, N]`` float32, ``a`` ``[M, K]``, ``b`` ``[N, K]`` exact E4M3 values in
    float32, K a multiple of 32) as the sm_90a FP8 WGMMA accumulates it, one ``m64nNk32`` e4m3 instruction per
    k32 step in K order, the result of each step the incoming accumulator of the next; modelled in float32
    exactly.  Per output element and per k32 step:
    the 32 products are exact and anchored at ``ea + eb`` of the two E4M3 encodings (``floor(log2 |x|)`` for a
    normal, -6 for a subnormal; zero products take no part); the step exponent ``E`` is the largest anchor or
    the incoming accumulator's true exponent (a zero accumulator takes no part); every product and the
    accumulator are truncated toward zero to multiples of ``2^(E - 13)``; their exact sum is truncated toward
    zero to 14 significant bits and stored in the float32 accumulator."""
    assert a.shape[-1] % WGMMA_K == 0, a.shape
    for start in range(0, a.shape[-1], WGMMA_K):
        a_s, b_s = a[:, start : start + WGMMA_K], b[:, start : start + WGMMA_K]
        prods = a_s.unsqueeze(-2) * b_s.unsqueeze(-3)  # [M, N, 32], exact
        e_prod = _e4m3_exponent(a_s).unsqueeze(-2) + _e4m3_exponent(b_s).unsqueeze(-3)
        e_max = torch.where(prods != 0, e_prod, torch.full_like(e_prod, -10_000)).amax(
            -1
        )
        _, e_acc = torch.frexp(acc.abs())
        e_acc = torch.where(
            acc != 0, e_acc.to(torch.int32) - 1, torch.full_like(e_max, -10_000)
        )
        e_max = torch.maximum(e_max, e_acc)
        g = _granularity(e_max, WGMMA_FRACTION_BITS, floor_log2)
        result = _truncate(prods, g.unsqueeze(-1)).sum(-1) + _truncate(acc, g)
        _, e_res = torch.frexp(result.abs())
        e_res = torch.where(
            result != 0,
            e_res.to(torch.int32) - 1,
            torch.zeros_like(e_res, dtype=torch.int32),
        )
        acc = _truncate(result, _granularity(e_res, WGMMA_RESULT_BITS - 1, floor_log2))
    return acc


def _fp8_wgmma_scores(q, k):
    """``S = q k^T`` (``q`` ``[M, K]``, ``k`` ``[N, K]``, exact E4M3 values in float32, K a multiple of 32) as
    the sm_90a FP8 WGMMA accumulates it from a zero accumulator (the kernels issue K = 128 as four ``m64nNk32``
    e4m3 WGMMAs, scale-d 0 on the first); every term is a multiple of the product unit ``2^-18``, where the
    granularity is floored."""
    acc = torch.zeros((q.shape[0], k.shape[0]), dtype=torch.float32, device=q.device)
    return _fp8_wgmma_accumulate(acc, q, k, floor_log2=-2 * E4M3_UNIT_LOG2)


def _fp8_wgmma_pv(acc, p, v):
    """``acc (+) p v`` of one position (``acc`` ``[M, 128]`` the chain's float32 O accumulator after the rescale,
    ``p`` ``[M, nk]`` the E4M3-rounded P as float32, ``v`` ``[nk, 128]`` exact E4M3 values in float32, ``nk`` a
    multiple of 32): the k32 steps in key order with the carried accumulator in every step's alignment set, no
    granularity floor.  A k32 step without a product (the persistent kernel's zero P half of a single-block
    position) leaves an accumulator of at most 14 significant bits unchanged, so the empty steps are not
    replayed."""
    return _fp8_wgmma_accumulate(acc, p, v.transpose(0, 1), floor_log2=-126)


def _torch_sage_quantize_hnd(q, k, v):
    """Torch Sage recipe on HND tensors (the SM100 recipe with ``batch := H`` and one head): E4M3 q/k/v and the
    FP32 scales in the plan's shapes ``(1, H, Sq)``, ``(1, H_kv, ceil(Sk / 16))``, ``(H_kv, 128)`` (``k`` / ``v``
    carry the kv heads of a GQA problem; every tensor is quantized per its own head)."""
    _h, sq, dim = q.shape
    h, sk = k.shape[0], k.shape[1]
    q32, k32, v32 = (t.float() for t in (q, k, v))
    q_scale = q32.abs().amax(dim=-1).clamp_min(1e-12) / FP8_MAX
    q8 = (q32 / q_scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    groups = -(-sk // K_GROUP)
    k_pad = torch.nn.functional.pad(k32, (0, 0, 0, groups * K_GROUP - sk))
    k_grouped = k_pad.view(h, groups, K_GROUP, dim)
    k_scale = k_grouped.abs().amax(dim=(-1, -2)).clamp_min(1e-12) / FP8_MAX
    k8 = (k_grouped / k_scale[..., None, None]).clamp(-FP8_MAX, FP8_MAX)
    k8 = k8.view(h, groups * K_GROUP, dim)[:, :sk].to(FP8)
    v_scale = v32.abs().amax(dim=-2).clamp_min(1e-12) / FP8_MAX
    v8 = (v32 / v_scale.unsqueeze(-2)).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    return (
        q8.contiguous(),
        k8.contiguous(),
        v8.contiguous(),
        q_scale.unsqueeze(0).contiguous(),
        k_scale.unsqueeze(0).contiguous(),
        v_scale.contiguous(),
    )


def _fp8_inputs(
    h,
    mb,
    nb,
    seed=7,
    qscale_mode="unit",
    kscale_mode="unit",
    vscale_mode="unit",
    quantize=None,
):
    """BF16 inputs of ``_inputs`` under the scale variants (peaked query rows, one key spike per head,
    per-channel V magnitudes), quantized with ``quantize`` (the torch recipe by default; the package quantizer
    for the kernel-vs-kernel checks)."""
    q, k, v = (t.clone() for t in _inputs(h, mb, nb, seed))
    if qscale_mode == "x3":  # peaked softmax: three times larger query rows
        q.mul_(3.0)
    if kscale_mode == "spike":  # one key per head, SPIKE_LOG2 octaves above the rest
        n = k.shape[1]
        for head in range(h):
            key = (n // 2 + 64 * head + 7) % n
            k[head, key] = (k[head, key].float() * 2.0**SPIKE_LOG2).bfloat16()
    if (
        vscale_mode == "dynamic"
    ):  # per-channel V magnitudes over V_DYNAMIC_LOG10 decades
        magnitudes = torch.logspace(*V_DYNAMIC_LOG10, 128, device="cuda")
        v = (v.float() * magnitudes).bfloat16()
    return (quantize or _torch_sage_quantize_hnd)(q, k, v)


def _dequantize(q8, k8, v8, q_scale, k_scale, v_scale):
    sk = k8.shape[1]
    k_tok = k_scale[0].repeat_interleave(K_GROUP, dim=-1)[:, :sk]
    return (
        q8.float() * q_scale[0].unsqueeze(-1),
        k8.float() * k_tok.unsqueeze(-1),
        v8.float() * v_scale.unsqueeze(-2),
    )


def _fp8_reference_exact(q8, k8, v8, scales, mask, scale):
    """Reference (b): exact FP32 softmax attention of the dequantized E4M3 inputs."""
    return _reference(*_dequantize(q8, k8, v8, *scales), mask, scale)


def _fp8_fma_rn(s, c, b):
    """``fma(s, c, b)`` of float32 tensors rounded once to float32, exactly, through float64: the product is
    exact in float64 (two 24-bit significands), the float64 sum ``hi`` carries its exact error ``lo`` (two-sum),
    and ``hi`` rounds to float32 as the exact sum does unless ``hi`` is exactly a float32 midpoint, where the
    sign of ``lo`` picks the side (ties to even only when ``lo == 0``)."""
    p = s.double() * c.double()
    bd = b.double()
    hi = p + bd
    bv = hi - p
    lo = (p - (hi - bv)) + (bd - bv)
    f = hi.float()
    back = f.double()
    toward = torch.where(
        hi > back, torch.full_like(f, float("inf")), torch.full_like(f, float("-inf"))
    )
    other = torch.nextafter(f, toward)
    midpoint = (back != hi) & (hi == (back + other.double()) / 2)
    beyond = midpoint & (lo != 0) & (torch.sign(lo) == torch.sign(hi - back))
    return torch.where(beyond, other, f)


def _fp8_sage_p(scores, qk_scale, m):
    """P x 2^FP8_P_QUANT_LOG2 of a step as the kernels form it: the argument ``fma(s, qk_sc, 8 - m)`` rounded
    once (PTX ``fma.rn.ftz.f32``; :func:`_fp8_fma_rn`) and its exponent by the device ``ex2.approx.ftz.f32``
    (``MUFU.EX2``), which ``torch.exp2`` of a CUDA float32 tensor reproduces bit for bit over the kernels'
    argument range (measured: 0 of 1.1e6 replayed arguments and 0 of 4.2e6 sweep points in [-30, 8.81]
    differ), its subnormal results flushed to zero as the instruction flushes them.  Three separate roundings
    of ``s qk_sc - m + 8`` move the argument by a few float32 ulps and P by up to about ten ulps, which
    crosses an E4M3 rounding boundary on a few elements per million."""
    bias = (FP8_P_QUANT_LOG2 - m)[:, None]
    s, c, b = torch.broadcast_tensors(scores.float(), qk_scale.float(), bias.float())
    x = _fp8_fma_rn(s, c, b)
    p = torch.exp2(x)
    return torch.where(x < -126.0, torch.zeros_like(p), p)


def _fp8_chain(
    qf_head,
    q_log2,
    kf_head,
    k_tok,
    vf_head,
    rows,
    steps,
    score_accumulate="fp8_wgmma",
    pv_accumulate="fp8_wgmma",
):
    """One lazy-max chain over ``steps`` (each a list of KV block ids processed as one softmax step) for the
    query rows ``rows``: returns ``(acc, m, l)`` in FP32 with P rounded to E4M3 before the P.V product.  The
    ``*_head`` tensors are the head's E4M3 values widened to FP32 (exact).  ``score_accumulate`` is how the
    scores of a step are accumulated: ``"fp8_wgmma"`` (the default, :func:`_fp8_wgmma_scores`) or ``"fp32"``
    (a float32 matmul, the opt-out); ``pv_accumulate`` how a step's P.V joins the chain's O accumulator:
    ``"fp8_wgmma"`` (the default, :func:`_fp8_wgmma_pv` after the rescale) or ``"fp32"`` (``acc + P.V`` in
    float32, the opt-out)."""
    if score_accumulate not in SCORE_ACCUMULATE_MODES:
        raise ValueError(
            f"score_accumulate must be one of {SCORE_ACCUMULATE_MODES}, got {score_accumulate!r}"
        )
    if pv_accumulate not in PV_ACCUMULATE_MODES:
        raise ValueError(
            f"pv_accumulate must be one of {PV_ACCUMULATE_MODES}, got {pv_accumulate!r}"
        )
    device = qf_head.device
    m = torch.full((len(rows),), float("-inf"), device=device)
    l = torch.zeros((len(rows),), device=device)
    acc = torch.zeros((len(rows), 128), device=device)
    qf = qf_head[rows]
    q_log2_rows = q_log2[rows]
    for blocks in steps:
        tok = torch.cat(
            [torch.arange(b * 64, (b + 1) * 64, device=device) for b in blocks]
        )
        if score_accumulate == "fp8_wgmma":
            scores = _fp8_wgmma_scores(qf, kf_head[tok])
        else:
            scores = qf @ kf_head[tok].T
        qk_scale = q_log2_rows[:, None] * k_tok[tok][None, :]
        s_log2 = scores * qk_scale
        step_max = s_log2.amax(-1)
        new_max = torch.maximum(m, step_max)
        first = m == float("-inf")
        grow = first | (new_max - m > FP8_MAX_GROWTH_LOG2)
        rescale = torch.where(
            grow & ~first, torch.exp2(m - new_max), torch.ones_like(m)
        )
        m = torch.where(grow, new_max, m)
        p = _fp8_sage_p(scores, qk_scale, m)
        p8 = p.to(FP8).float()
        if pv_accumulate == "fp8_wgmma":
            acc = _fp8_wgmma_pv(acc * rescale[:, None], p8, vf_head[tok])
        else:
            acc = acc * rescale[:, None] + p8 @ vf_head[tok]
        l = l * rescale + p.sum(-1)
    return acc, m, l


def _fp8_fold(parts):
    """Flash-decoding fold of ``(acc, m, l)`` chains in the given (fixed) order with the weights
    ``exp2(m_j - M)``; a chain without a step (``m = -inf``) carries no weight."""
    m_all = parts[0][1]
    for _, m, _ in parts[1:]:
        m_all = torch.maximum(m_all, m)
    acc = torch.zeros_like(parts[0][0])
    l = torch.zeros_like(m_all)
    for part_acc, m, part_l in parts:
        w = torch.where(m > float("-inf"), torch.exp2(m - m_all), torch.zeros_like(m))
        acc = acc + part_acc * w[:, None]
        l = l + part_l * w
    return acc, m_all, l


def _fp8_reference_replay(
    plan_dict,
    kind,
    q8,
    k8,
    v8,
    scales,
    mask,
    scale,
    score_accumulate="fp8_wgmma",
    pv_accumulate="fp8_wgmma",
):
    """Reference (a): the plan-faithful replay, BF16 ``[H, M, 128]`` like the kernel's output.  ``plan_dict``
    is the planner's own dict for the plan object's route (``plan_vsa_sm90`` or ``plan_small``; the FP8 plan
    consumes the same bytes), ``kind`` ``"persistent"`` or ``"small"``.  Persistent: per tile and warpgroup,
    the ordered own positions (block pairs) of ``_decode_tiles`` form one chain (a tagged plan's single-reader
    tag bit stripped); a split tile's two chains fold WG1 into WG0.  Small: per item the blocks alternate
    between the two warpgroups by position parity (``kmax`` 1: one warpgroup), the WG1 chain folds into WG0;
    split-KV / cluster slices fold in slice order, a split-cluster row's balanced slices derived as the kernel
    does (``_balanced_chunks``).  The scores and the P.V accumulate follow ``score_accumulate`` /
    ``pv_accumulate`` (see :func:`_fp8_chain`)."""
    q_scale, k_scale, v_scale = scales
    h, mb, _ = mask.shape
    sk = k8.shape[1]
    out = torch.empty((h, mb * 64, 128), device=q8.device, dtype=torch.bfloat16)
    k_tok = k_scale[0].repeat_interleave(K_GROUP, dim=-1)[:, :sk]
    q_log2 = q_scale[0] * (float(scale) * LOG2E)
    widened: dict[int, tuple] = {}

    def head_values(head):
        if head not in widened:
            widened[head] = (q8[head].float(), k8[head].float(), v8[head].float())
        return widened[head]

    def chain(head, qb, steps):
        rows = torch.arange(qb * 64, (qb + 1) * 64, device=q8.device)
        qf, kf, vf = head_values(head)
        return _fp8_chain(
            qf,
            q_log2[head],
            kf,
            k_tok[head],
            vf,
            rows,
            steps,
            score_accumulate,
            pv_accumulate,
        )

    folded: dict[tuple[int, int], list] = {}
    if kind == "persistent":
        tag = TAG if plan_dict["tagged"] else 0
        for head, qb0, qb1, mode, positions, owns in _decode_tiles(plan_dict):
            for wg, entries in enumerate(owns):
                if mode in (MODE_HEAD_PAIR, MODE_HEAD_PAIR_SPLIT):
                    # a grouped-query plan's head-pair / re-paired tile: warpgroup w serves head + w at qb0
                    key = (head + wg, qb0)
                else:
                    key = (head, qb0 if (mode or wg == 0) else qb1)
                steps = []
                for entry in entries:
                    a, b = positions[entry >> 3]
                    steps.append([a & ~tag] if b < 0 else [a & ~tag, b & ~tag])
                if not steps:
                    continue
                folded.setdefault(key, []).append(chain(*key, steps))
    else:
        kmax = plan_dict["kmax"]
        split_cluster = bool(plan_dict["split"] and plan_dict["cluster"])

        def item(head, qb, blocks):
            """One plan item: the blocks alternate by position parity between the two warpgroups (the
            kernel's BLOCK_PARITY), WG1 folded into WG0; ``kmax`` 1 runs one warpgroup."""
            parities = (0,) if kmax == 1 else (0, 1)
            chains = []
            for parity in parities:
                steps = [[b] for i, b in enumerate(blocks) if i % 2 == parity]
                if steps:
                    chains.append(chain(head, qb, steps))
            return _fp8_fold(chains)

        slices: dict[tuple[int, int], list] = {}
        for row in _decode_small_rows(plan_dict):
            tile = row["tile"]
            head, qb = divmod(tile, mb)
            if not row["blocks"]:
                continue
            if split_cluster:
                # one row per query block; the kernel's cluster derives the balanced slices from the count
                nsplit = max(1, -(-row["count"] // kmax))
                for j, ids in enumerate(
                    planner._balanced_chunks(row["blocks"], nsplit)
                ):
                    slices.setdefault((head, qb), []).append((j, item(head, qb, ids)))
            else:
                slices.setdefault((head, qb), []).append(
                    (row["split"], item(head, qb, row["blocks"]))
                )
        for key, parts in slices.items():
            folded[key] = [part for _, part in sorted(parts, key=lambda item: item[0])]
    for (head, qb), parts in folded.items():
        acc, _, l = _fp8_fold(parts)
        out[head, qb * 64 : (qb + 1) * 64] = (
            acc * v_scale[head][None, :] / l[:, None]
        ).bfloat16()
    return out


def _fp8_plan_dict(plan, mask, mask_kv=None):
    """The planner dict of the route the plan object took (same planner, same bytes).  ``mask_kv`` (the
    kv-head mask of a grouped-query plan) selects the GQA planner's own tiles -- head-pair and re-paired
    tiles included -- instead of the MHA plan of the replicated ``mask``."""
    if plan.small_kmax is None:
        sms = torch.cuda.get_device_properties(plan.device).multi_processor_count
        if mask_kv is not None and plan.gqa > 1:
            return "persistent", plan_vsa_sm90(
                mask_kv.cpu(), sms=sms, gqa=plan.gqa, split_tile_order="group"
            )
        return "persistent", plan_vsa_sm90(mask.cpu(), sms=sms)
    return "small", plan_small(
        mask.cpu(),
        kmax=plan.small_kmax,
        split=plan.small_split,
        cluster=plan.small_cluster,
    )


def _fp8_check(
    plan,
    out,
    q8,
    k8,
    v8,
    scales,
    mask,
    scale,
    score_accumulate="fp8_wgmma",
    pv_accumulate="fp8_wgmma",
    mask_kv=None,
):
    """Both references on one output: (a) the plan-faithful replay at atol = rtol = 1e-2, (b) the exact softmax
    of the dequantized inputs at atol = rtol = 5e-2, no element outside.  ``mask_kv``: replay the grouped-query
    plan's own tiles (see :func:`_fp8_plan_dict`)."""
    kind, plan_dict = _fp8_plan_dict(plan, mask, mask_kv)
    replay = _fp8_reference_replay(
        plan_dict,
        kind,
        q8,
        k8,
        v8,
        scales,
        mask,
        scale,
        score_accumulate,
        pv_accumulate,
    )
    torch.testing.assert_close(out.float(), replay.float(), atol=1e-2, rtol=1e-2)
    exact = _fp8_reference_exact(q8, k8, v8, scales, mask, scale)
    torch.testing.assert_close(out.float(), exact, atol=5e-2, rtol=5e-2)


def _fp8_plan_for(mask, scales, scale=None, route=None):
    h, mb, nb = mask.shape
    rows = torch.full((h, mb), 64, dtype=torch.int32, device="cuda")
    cols = torch.full((h, nb), 64, dtype=torch.int32, device="cuda")
    return CakeVsaSm90Plan(
        "cuda",
        mask,
        rows,
        cols,
        h,
        h,
        128,
        sm_scale=scale,
        q_data_type=FP8,
        q_scale=scales[0],
        k_scale=scales[1],
        v_scale=scales[2],
        route=route,
    )


def _profiled_kernel_names(fn, scheduled: bool) -> list[str]:
    """CUDA kernel names of one call of ``fn`` from a ``torch.profiler`` session (memcpy / memset activities
    excluded); the scheduled form runs one warm-up call whose records are discarded."""
    kwargs = {"activities": [torch.profiler.ProfilerActivity.CUDA]}
    if scheduled:
        kwargs["schedule"] = torch.profiler.schedule(
            wait=0, warmup=1, active=1, repeat=1
        )
    with torch.profiler.profile(**kwargs) as profile:
        fn()
        torch.cuda.synchronize()
        if scheduled:
            profile.step()
            fn()
            torch.cuda.synchronize()
    return [
        event.name
        for event in profile.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
        and not event.name.startswith(("Memcpy", "Memset"))
    ]


def _graph_kernel_names(fn) -> list[str]:
    """Kernel names of one call of ``fn`` read from a CUDA graph capture of it through cuda-python (memset /
    memcpy nodes excluded): a CUPTI-free route for a process whose profiler no longer records device activity."""
    from cuda.bindings import driver

    def check(result):
        err, *rest = result
        assert err == driver.CUresult.CUDA_SUCCESS, err
        return rest[0] if len(rest) == 1 else rest

    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    handle = driver.CUgraph(graph.raw_cuda_graph())
    _, count = check(driver.cuGraphGetNodes(handle, 0))
    nodes, count = check(driver.cuGraphGetNodes(handle, count))
    names = []
    for node in nodes[:count]:
        kind = check(driver.cuGraphNodeGetType(node))
        if kind != driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            continue
        params = check(driver.cuGraphKernelNodeGetParams(node))
        func = params.func
        if int(func) == 0 and int(params.kern) != 0:
            func = check(driver.cuKernelGetFunction(params.kern))
        names.append(check(driver.cuFuncGetName(func)).decode())
    return names


def _cuda_kernel_names(fn) -> list[str]:
    """CUDA kernel names of one call of ``fn``, read from a CUDA graph capture of it (every kernel node, in launch
    order; memset / memcpy nodes excluded).  The capture route does not depend on CUPTI: in a long test process
    Kineto drops device records whose GPU timestamps fall outside the profiler's capture window, so a
    ``torch.profiler`` session can come back empty or partial.  The profiler route is kept only for a torch
    without ``CUDAGraph.raw_cuda_graph``."""
    if hasattr(torch.cuda.CUDAGraph, "raw_cuda_graph"):
        return _graph_kernel_names(fn)
    return _profiled_kernel_names(fn, scheduled=True)


@requires_hopper
@pytest.mark.parametrize(
    "h,mb,nb,capacity,scale",
    [
        (1, 1, 1, 1, None),
        (4, 4, 4, 1, 0.5),
        (8, 4, 4, 3, None),
        (8, 16, 16, 4, -0.125),
        (2, 8, 32, 6, 0.0),
    ],
)
def test_fp8_small_and_persistent_routes_agree(h, mb, nb, capacity, scale):
    """Both FP8 kernels implement the same contract: force each route on a small problem, check both
    references, and require three launches to be bit-identical."""
    mask = _random_mask(h, mb, nb, capacity, seed=11, ragged=True, device="cuda")
    q8, k8, v8, *scales = _fp8_inputs(h, mb, nb)
    sm_scale = 128**-0.5 if scale is None else scale
    for route in ("small", "persistent"):
        plan = _fp8_plan_for(mask, scales, scale, route=route)
        assert plan.fp8 and plan.stage.startswith(
            "small_fp8_" if route == "small" else "attention_fp8"
        )
        outs = [plan.run(q8, k8, v8).clone() for _ in range(3)]
        assert outs[0].dtype == torch.bfloat16
        for later in outs[1:]:
            torch.testing.assert_close(later, outs[0], atol=0, rtol=0)
        _fp8_check(plan, outs[0], q8, k8, v8, scales, mask, sm_scale)
    auto = _fp8_plan_for(mask, scales, scale)
    assert auto.mode == "small" and auto.stage.startswith("small_fp8_")


@requires_hopper
@pytest.mark.parametrize(
    "h,mb,nb,capacity,scale,ragged",
    [
        (1, 16, 16, 16, None, False),
        (1, 16, 64, 16, 0.5, True),
        (2, 8, 64, 64, None, True),  # 64-block lists: the longest tails
        (1, 1, 8, 6, 1e-7, False),
        (3, 5, 10, 9, 0.0, True),
        (3, 5, 10, 9, -0.125, True),
    ],
)
def test_fp8_split_route_against_persistent(h, mb, nb, capacity, scale, ragged):
    """The FP8 split-KV kernel: two runs bit-identical (atol 0), counters back at zero, both references."""
    mask = _random_mask(h, mb, nb, capacity, seed=13, ragged=ragged, device="cuda")
    q8, k8, v8, *scales = _fp8_inputs(h, mb, nb)
    sm_scale = 128**-0.5 if scale is None else scale
    split = _fp8_plan_for(mask, scales, scale, route="smallsplit")
    assert (
        split.mode == "smallsplit"
        and split.small_split
        and split.stage.startswith("small_fp8_k")
    )
    first = split.run(q8, k8, v8).clone()
    second = split.run(q8, k8, v8)
    torch.cuda.synchronize()
    assert int(split.counters.abs().sum()) == 0
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    _fp8_check(split, first, q8, k8, v8, scales, mask, sm_scale)
    persistent = _fp8_plan_for(mask, scales, scale, route="persistent")
    _fp8_check(
        persistent, persistent.run(q8, k8, v8), q8, k8, v8, scales, mask, sm_scale
    )
    auto = _fp8_plan_for(mask, scales, scale)
    rule = small_route(
        mask,
        sms=torch.cuda.get_device_properties(0).multi_processor_count,
        cluster_capacity=cluster_capacity(torch.cuda.current_device()),
    )
    assert auto.mode == _mode_of(rule)


@requires_hopper
@pytest.mark.parametrize(
    "h,mb,nb,capacity,scale,ragged,route",
    [
        (1, 16, 16, 16, None, False, "smallcluster"),
        (4, 16, 16, 8, 0.5, True, "smallcluster"),
        (1, 16, 64, 24, None, False, "smallsplitcluster"),
        (2, 8, 64, 36, -0.125, True, "smallsplitcluster"),
    ],
)
def test_fp8_cluster_routes_against_persistent(
    h, mb, nb, capacity, scale, ragged, route
):
    """The FP8 DSM-cluster and split-cluster kernels: two runs bit-identical, counters zero, both references."""
    mask = _random_mask(h, mb, nb, capacity, seed=19, ragged=ragged, device="cuda")
    q8, k8, v8, *scales = _fp8_inputs(h, mb, nb)
    sm_scale = 128**-0.5 if scale is None else scale
    plan = _fp8_plan_for(mask, scales, scale, route=route)
    assert (
        plan.mode == route
        and plan.small_cluster > 0
        and plan.stage.startswith("small_fp8_k")
    )
    first = plan.run(q8, k8, v8).clone()
    torch.testing.assert_close(plan.run(q8, k8, v8), first, atol=0, rtol=0)
    if plan.small_split:
        torch.cuda.synchronize()
        assert int(plan.counters.abs().sum()) == 0
    _fp8_check(plan, first, q8, k8, v8, scales, mask, sm_scale)
    persistent = _fp8_plan_for(mask, scales, scale, route="persistent")
    _fp8_check(
        persistent, persistent.run(q8, k8, v8), q8, k8, v8, scales, mask, sm_scale
    )


@requires_hopper
@pytest.mark.parametrize("variant", ["x3", "spike", "dynamic", "zero_scale"])
def test_fp8_scale_variants(variant):
    """The scale variants on a ragged pair plan: peaked softmax (Q rows x3), one key
    per head two octaves above the rest (the growth branch of the lazy maximum mid-chain), per-channel V
    magnitudes over three decades (the ``v_scale`` path), and ``sm_scale = 0`` (uniform softmax)."""
    h, mb, nb = 4, 16, 16
    mask = _random_mask(h, mb, nb, 8, seed=23, ragged=True, device="cuda")
    modes = {
        "x3": {"qscale_mode": "x3"},
        "spike": {"kscale_mode": "spike"},
        "dynamic": {"vscale_mode": "dynamic"},
        "zero_scale": {},
    }[variant]
    scale = 0.0 if variant == "zero_scale" else None
    q8, k8, v8, *scales = _fp8_inputs(h, mb, nb, **modes)
    sm_scale = 128**-0.5 if scale is None else scale
    for route in ("persistent", None):
        plan = _fp8_plan_for(mask, scales, scale, route=route)
        _fp8_check(plan, plan.run(q8, k8, v8), q8, k8, v8, scales, mask, sm_scale)


@requires_hopper
def test_fp8_empty_rows_raise_the_bf16_error():
    """An empty query block is refused with the same ValueError text as the BF16 plan, on the automatic
    route and on the forced persistent route (``plan_vsa_sm90``'s message)."""
    mask = _random_mask(2, 4, 8, 3, seed=3, ragged=True, device="cuda")
    mask[1, 2] = False
    q8, k8, v8, *scales = _fp8_inputs(2, 4, 8)
    rows = torch.full((2, 4), 64, dtype=torch.int32, device="cuda")
    cols = torch.full((2, 8), 64, dtype=torch.int32, device="cuda")
    for route in (None, "persistent"):
        messages = {}
        for dtype in (torch.bfloat16, FP8):
            kwargs = (
                {"q_scale": scales[0], "k_scale": scales[1], "v_scale": scales[2]}
                if dtype is FP8
                else {}
            )
            with pytest.raises(ValueError) as info:
                CakeVsaSm90Plan(
                    "cuda",
                    mask,
                    rows,
                    cols,
                    2,
                    2,
                    128,
                    q_data_type=dtype,
                    route=route,
                    **kwargs,
                )
            messages[dtype] = str(info.value)
        assert messages[torch.bfloat16] == messages[FP8]
        if route == "persistent":
            assert messages[FP8] == "empty sparse rows are not supported"


@requires_hopper
@pytest.mark.parametrize(
    "route", ["persistent", "smallsplit", "smallcluster", "smallsplitcluster"]
)
def test_fp8_plan_stream_and_graph_lifetime(route):
    """An FP8 plan built on another stream, captured and replayed bit-exactly into a NaN-prefilled output
    (every element written), counters zero after every launch; scales are plan data, so quantizing new inputs
    into the captured tensors in place is what the replay computes."""
    h, mb, nb = 1, 16, 64 if "split" in route else 16
    capacity = {
        "persistent": 16,
        "smallsplit": 16,
        "smallcluster": 16,
        "smallsplitcluster": 24,
    }[route]
    mask = _random_mask(h, mb, nb, capacity, seed=17, ragged=False, device="cuda")
    q, k, v = (t.clone() for t in _inputs(h, mb, nb))
    q8, k8, v8, *scales = _torch_sage_quantize_hnd(q, k, v)
    producer = torch.cuda.Stream()
    with torch.cuda.stream(producer):
        plan = _fp8_plan_for(mask, scales, route=route)
    # a persistent plan's ``mode`` is the planner's tile mode (pair / split), the small routes' their route name
    if route == "persistent":
        assert (
            plan.small_kmax is None
            and plan.stage.startswith("attention_fp8")
            and plan.fp8
        )
    else:
        assert plan.mode == route and plan.fp8
    expected = plan.run(q8, k8, v8).clone()
    out = torch.full(
        (h * mb * 64, 1, 128), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    plan.run(q8, k8, v8, out=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan.run(q8, k8, v8, out=out)
    for _ in range(3):
        out.fill_(float("nan"))
        graph.replay()
    torch.cuda.synchronize()
    assert not torch.isnan(out).any()
    torch.testing.assert_close(out.view_as(expected), expected, atol=0, rtol=0)
    if plan.small_split:
        assert int(plan.counters.abs().sum()) == 0
    q.normal_()
    fresh = _torch_sage_quantize_hnd(q, k, v)
    for captured, new in zip((q8, k8, v8, *scales), fresh, strict=True):
        captured.copy_(new)
    graph.replay()
    _fp8_check(plan, out.view_as(expected), q8, k8, v8, scales, mask, 128**-0.5)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="cake (SM90) VSA requires Hopper compute capability 9.0",
)
def test_fp8_queue_kernel_matches_the_static_kernel_bit_exact(monkeypatch):
    """The FP8 queue kernel only moves tiles between CTAs: bit-identical to the FP8 static kernel on the same
    plan, queue words back at zero, graph replays bit-identical."""
    h, mb, nb = 8, 528, 64
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    mask = _topk_mask(h, mb, nb, 8).cuda()
    q8, k8, v8, *scales = _fp8_inputs(h, mb, nb)
    queued = _fp8_plan_for(mask, scales, route="persistent")
    if not queued.queue:
        pytest.skip(f"{sms} SMs: fewer than {POOL_MIN_TILES} tiles per CTA")
    assert queued.stage == "attention_fp8_queue"
    out = torch.empty((h * mb * 64, 1, 128), device="cuda", dtype=torch.bfloat16)
    first = queued.run(q8, k8, v8, out=out).clone()
    torch.cuda.synchronize()
    assert int(queued.dbg[62:64].abs().sum()) == 0
    monkeypatch.setattr(planner, "POOL_MIN_TILES", 10**9)
    static = _fp8_plan_for(mask, scales, route="persistent")
    assert (
        not static.queue
        and static.stage == "attention_fp8"
        and static.num_ctas == queued.num_ctas
    )
    torch.testing.assert_close(static.run(q8, k8, v8), first, atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        queued.run(q8, k8, v8, out=out)
    for _ in range(3):
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out.view_as(first), first, atol=0, rtol=0)
        assert int(queued.dbg[62:64].abs().sum()) == 0
    exact = _fp8_reference_exact(q8, k8, v8, scales, mask, 128**-0.5)
    torch.testing.assert_close(first.float(), exact, atol=5e-2, rtol=5e-2)


@requires_hopper
@pytest.mark.parametrize(
    "mutation,match",
    [
        ("missing_k", r"q_scale, k_scale and v_scale together \(missing: k_scale\)"),
        ("missing_all", r"together \(missing: q_scale, k_scale, v_scale\)"),
        ("bf16_with_scales", r"only with q_data_type=float8_e4m3fn"),
        ("kv_dtype", r"kv_data_type equal to q_data_type"),
        (
            "q_shape",
            r"q_scale must have shape \(1, 2, 256\) and dtype float32, got \(2, 256\) float32",
        ),
        (
            "k_groups",
            r"k_scale must have shape \(1, 2, 32\) and dtype float32, got \(1, 2, 31\) float32",
        ),
        (
            "v_shape",
            r"v_scale must have shape \(2, 128\) and dtype float32, got \(2, 1, 128\) float32",
        ),
        (
            "k_dtype",
            r"k_scale must have shape \(1, 2, 32\) and dtype float32, got \(1, 2, 32\) bfloat16",
        ),
        ("q_strided", r"q_scale must be contiguous on the planned device"),
        # GQA: the K scale is per kv head (a query-head-shaped one is refused by its shape) and the head ratio
        # must be in {2, 4, 8}
        (
            "gqa_k_scale_at_query_heads",
            r"k_scale must have shape \(1, 1, 32\) and dtype float32, got \(1, 2, 32\) float32",
        ),
        ("gqa_ratio_3", r"g in \(2, 4, 8\)"),
    ],
)
def test_fp8_plan_rejects_invalid_scales(mutation, match):
    """The plan object's own validation of the Sage FP8 inputs (exact messages the wrapper forwards); a GQA
    request is validated against the kv-head-shaped mask and scales."""
    h, mb, nb = 2, 4, 8  # Sq 256, Sk 512 -> 32 groups
    mask = _random_mask(h, mb, nb, 3, seed=3, ragged=True, device="cuda")
    rows = torch.full((h, mb), 64, dtype=torch.int32, device="cuda")
    cols = torch.full((h, nb), 64, dtype=torch.int32, device="cuda")
    q8, k8, v8, q_scale, k_scale, v_scale = _fp8_inputs(h, mb, nb)
    kwargs = {
        "q_data_type": FP8,
        "q_scale": q_scale,
        "k_scale": k_scale,
        "v_scale": v_scale,
    }
    kv_h = h
    valid = (mask, rows, cols)
    if mutation == "missing_k":
        del kwargs["k_scale"]
    elif mutation == "missing_all":
        kwargs = {"q_data_type": FP8}
    elif mutation == "bf16_with_scales":
        kwargs["q_data_type"] = torch.bfloat16
    elif mutation == "kv_dtype":
        kwargs["kv_data_type"] = torch.bfloat16
    elif mutation == "q_shape":
        kwargs["q_scale"] = q_scale[0]
    elif mutation == "k_groups":
        kwargs["k_scale"] = k_scale[..., :31]
    elif mutation == "v_shape":
        kwargs["v_scale"] = v_scale.unsqueeze(1)
    elif mutation == "k_dtype":
        kwargs["k_scale"] = k_scale.bfloat16()
    elif mutation == "q_strided":
        kwargs["q_scale"] = torch.ones((1, 2, 512), device="cuda")[..., ::2]
    elif mutation == "gqa_k_scale_at_query_heads":
        # two query heads over one kv head: the mask and the K / V scales are per kv head
        mask, rows, cols = mask[:1], rows[:1], cols[:1]
        kv_h = 1
        kwargs["v_scale"] = v_scale[:1]
    else:
        # six query heads over two kv heads
        kwargs["q_scale"] = torch.ones((1, 6, 256), device="cuda")
        h = 6
    with pytest.raises(ValueError, match=match):
        CakeVsaSm90Plan("cuda", mask, rows, cols, h, kv_h, 128, **kwargs)
    h = 2
    mask, rows, cols = valid
    # the valid request plans
    plan = CakeVsaSm90Plan(
        "cuda",
        mask,
        rows,
        cols,
        h,
        h,
        128,
        q_data_type=FP8,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    assert plan.fp8 and plan.k_scale_groups == 32
    for name in ("q", "k", "v"):
        with pytest.raises(
            ValueError, match=f"{name} must have shape .* and dtype float8_e4m3fn"
        ):
            plan.run(
                *[
                    t.view(torch.bfloat16) if n == name else t
                    for n, t in zip("qkv", (q8, k8, v8), strict=True)
                ]
            )


@requires_hopper
@pytest.mark.parametrize(
    "shape", [(1, 256, 256), (4, 1000, 1500), (3, 65, 4000), (8, 4096, 4096)]
)
def test_sage_fp8_quantize_sm90_matches_torch(shape, record_property):
    """The package quantizer against the torch Sage recipe on HND tensors: the SM100 quantizer's criterion
    (scales within 1e-5 relative -- the kernel multiplies by ``1/448`` where torch divides -- and values within
    one E4M3 step with zero violations), the exact output shapes ``plan()`` takes, exactly two kernels behind
    one fill, and the bitwise mismatch counts reported (bitwise identity is not required: the kernel floors
    ``amax`` at 1e-6 and uses the reciprocal)."""
    h, sq, sk = shape
    g = torch.Generator(device="cuda").manual_seed(sq * 7 + sk)
    q = torch.randn((h, sq, 128), device="cuda", generator=g).bfloat16()
    k = torch.randn((h, sk, 128), device="cuda", generator=g).bfloat16()
    v = torch.randn((h, sk, 128), device="cuda", generator=g).bfloat16()
    got = pkg.sage_fp8_quantize_sm90(q, k, v)
    torch.cuda.synchronize()
    ref = _torch_sage_quantize_hnd(q, k, v)
    groups = -(-sk // 16)
    assert [tuple(t.shape) for t in got[3:]] == [(1, h, sq), (1, h, groups), (h, 128)]
    assert all(t.dtype == torch.float32 and t.is_contiguous() for t in got[3:])
    for name, a, b in zip(
        ("q_scale", "k_scale", "v_scale"), got[3:], ref[3:], strict=True
    ):
        assert a.shape == b.shape, name
        rel = float((a - b).abs().max() / b.abs().max())
        assert rel < 1e-5, f"{name}: max relative scale error {rel:.3e}"
        record_property(f"bitwise_mismatch_{name}", int((a != b).sum()))
    for name, a, b in zip(("q8", "k8", "v8"), got[:3], ref[:3], strict=True):
        assert a.dtype == FP8 and a.shape == b.shape, name
        af, bf = a.float(), b.float()
        step = torch.maximum(af.abs(), bf.abs()).clamp_min(1.0) * 0.125
        mismatch = int(((af - bf).abs() > step).sum())
        assert mismatch == 0, (
            f"{name}: {mismatch} elements differ by more than one E4M3 step"
        )
        record_property(
            f"bitwise_mismatch_{name}",
            int((a.view(torch.uint8) != b.view(torch.uint8)).sum()),
        )
    names = _cuda_kernel_names(lambda: pkg.sage_fp8_quantize_sm90(q, k, v))
    kernels = [name for name in names if name.startswith("kernel_")]
    assert kernels == [
        loader.MODULES["quantize_qk_vamax"]["kernel_symbol"],
        loader.MODULES["quantize_v"]["kernel_symbol"],
    ], names
    assert len(names) == len(kernels) + 1 and "Fill" in names[0], names


@requires_hopper
def test_sage_fp8_quantize_sm90_rejects_other_layouts():
    q = torch.randn((2, 64, 128), device="cuda").bfloat16()
    k = torch.randn((2, 96, 128), device="cuda").bfloat16()
    with pytest.raises(ValueError, match="rank-3 HND tensor"):
        pkg.sage_fp8_quantize_sm90(q.unsqueeze(0), k, k)
    with pytest.raises(ValueError, match="head_dim must be 128"):
        pkg.sage_fp8_quantize_sm90(q[..., :64].contiguous(), k, k)
    with pytest.raises(
        ValueError, match=r"v must have shape \(2, 96, 128\) and dtype bfloat16"
    ):
        pkg.sage_fp8_quantize_sm90(q, k, k.float())
    # k / v with the kv heads of a GQA problem are accepted (g in {2, 4, 8}); other head counts are refused
    with pytest.raises(ValueError, match=r"v must have shape \(2, 96, 128\)"):
        pkg.sage_fp8_quantize_sm90(q, k, k[:1])
    with pytest.raises(ValueError, match=r"g in \(2, 4, 8\)"):
        pkg.sage_fp8_quantize_sm90(torch.cat([q, q[:1]]), k[:1], k[:1])
    with pytest.raises(ValueError, match="multiple of H_kv"):
        pkg.sage_fp8_quantize_sm90(torch.cat([q, q[:1]]), k, k)
    with pytest.raises(ValueError, match="q must be contiguous"):
        pkg.sage_fp8_quantize_sm90(q.transpose(0, 1).contiguous().transpose(0, 1), k, k)


@requires_hopper
def test_fp8_kernel_quantizer_feeds_the_fp8_route():
    """End to end inside the package: the kernel quantizer's outputs plan and run, and the output matches the
    exact softmax of what the kernel quantizer produced (reference (b)) and the plan-faithful replay (a)."""
    h, mb, nb = 2, 8, 16
    mask = _random_mask(h, mb, nb, 6, seed=29, ragged=True, device="cuda")
    q8, k8, v8, *scales = _fp8_inputs(h, mb, nb, quantize=pkg.sage_fp8_quantize_sm90)
    plan = _fp8_plan_for(mask, scales)
    _fp8_check(plan, plan.run(q8, k8, v8), q8, k8, v8, scales, mask, 128**-0.5)


# --------------------------------------------------------------------------- GQA route (H100; the GQA stages)


def _gqa_inputs(h, h_kv, mb, nb, seed=7):
    """BF16 ``q`` over the ``h`` query heads and ``k`` / ``v`` over the ``h_kv`` kv heads: ``q`` is ``_inputs``'
    at ``h`` heads, ``k`` / ``v`` are ``_inputs``' at ``h_kv`` heads (the draw an ``h_kv``-head MHA problem
    gets, so the MHA route on ``k.repeat_interleave(g, 0)`` is the replicated-input baseline)."""
    q = _inputs(h, mb, nb, seed)[0]
    _, k, v = _inputs(h_kv, mb, nb, seed)
    return q, k, v


def _descriptors_of(mask):
    h, mb, nb = mask.shape
    rows = torch.full((h, mb), 64, dtype=torch.int32, device=mask.device)
    cols = torch.full((h, nb), 64, dtype=torch.int32, device=mask.device)
    return rows, cols


def _hot_and_private_mask(h, mb, hot, private):
    """Every query block of a head selects the head's first ``hot`` blocks (read by every tile of the head)
    and ``private`` blocks of its own (read by one tile only): ``nb = hot + mb x private`` blocks, ``hot +
    private`` per row.  At g = 2 the pair plan's K / V^T footprint exceeds the L2 budget while the re-read
    (hot) blocks fit it, so the planner tags the single-reader blocks and takes the tagged program."""
    nb = hot + mb * private
    mask = torch.zeros((h, mb, nb), dtype=torch.bool)
    mask[:, :, :hot] = True
    for qb in range(mb):
        mask[:, qb, hot + qb * private : hot + (qb + 1) * private] = True
    return mask


def _static_route(plan):
    """Whether the plan's program is one whose GQA output must equal the MHA route's bit for bit on
    replicated inputs: every persistent program (static tile lists; split-mode tiles stay MHA-shaped unless
    the planner re-paired them; the queue program equals the static one) and the unsplit small programs.  The
    split / cluster small programs merge partials and are held to the route gate (their error against the
    reference may not exceed the MHA route's); their bitwise mismatch count is recorded.  A re-paired
    persistent plan (``repair_split``: one chain over the whole KV list per query head instead of the fold of
    two parity chains) is held to the route's absolute gates only; its ratios to the MHA route are recorded."""
    if plan.small_kmax is None:
        return not plan.repair_split
    return not (plan.small_split or plan.small_cluster)


def _error_stats(out, reference):
    """RMS and max-abs error of a BF16 output against the FP32 reference."""
    err = out.float() - reference
    return float(err.square().mean().sqrt()), float(err.abs().max())


def _check_gqa_against_mha(
    gqa_plan, gqa_out, mha_out, reference, record_property, label
):
    """The GQA output against the MHA route's on replicated inputs.  Recorded for every route: the bitwise
    mismatch count and both outputs' RMS / max-abs error against the FP32 reference.  Asserted: the route
    gate -- the GQA program's error may not exceed the MHA route's (RMS within 1.05x, max-abs within
    max(1.10x, 1e-3)) -- and, on the static programs, bitwise identity.  A re-paired persistent plan
    (``repair_split``, the planner of record's "auto" on uniform multi-wave split plans) is exempt from both:
    its ratios are recorded and the caller's absolute gates (the references at their tolerances) decide."""
    mismatch = int((gqa_out.view(torch.int16) != mha_out.view(torch.int16)).sum())
    rms_gqa, max_gqa = _error_stats(gqa_out, reference)
    rms_mha, max_mha = _error_stats(mha_out, reference)
    repaired = bool(getattr(gqa_plan, "repair_split", False))
    record_property(f"gqa_repair_split_{label}", repaired)
    record_property(f"gqa_vs_mha_bitwise_mismatch_{label}", mismatch)
    record_property(f"gqa_rms_{label}", f"{rms_gqa:.4e}")
    record_property(f"mha_rms_{label}", f"{rms_mha:.4e}")
    record_property(
        f"gqa_vs_mha_rms_ratio_{label}",
        f"{rms_gqa / rms_mha:.4f}" if rms_mha > 0 else "n/a",
    )
    record_property(f"gqa_max_abs_{label}", f"{max_gqa:.4e}")
    record_property(f"mha_max_abs_{label}", f"{max_mha:.4e}")
    if repaired:
        record_property(
            f"gqa_vs_mha_max_abs_ratio_{label}",
            f"{max_gqa / max_mha:.4f}" if max_mha > 0 else "n/a",
        )
        return
    assert rms_gqa <= 1.05 * rms_mha, (
        f"{label}: RMS error {rms_gqa:.4e} exceeds 1.05 x the MHA route's {rms_mha:.4e}"
    )
    assert max_gqa <= max(1.10 * max_mha, 1e-3), (
        f"{label}: max-abs error {max_gqa:.4e} exceeds max(1.10 x the MHA route's "
        f"{max_mha:.4e}, 1e-3)"
    )
    if _static_route(gqa_plan):
        assert mismatch == 0, (
            f"{label}: {mismatch} output elements differ from the MHA route on replicated inputs"
        )


@requires_hopper
@pytest.mark.parametrize(
    "g,h_kv,mb,nb,capacity,ragged,scale,route",
    [
        (2, 3, 4, 8, 3, True, None, None),  # small_gqa_k3, odd kv-head count
        (4, 2, 4, 8, 3, True, -0.125, "small"),
        (
            8,
            1,
            4,
            4,
            1,
            False,
            None,
            "small",
        ),  # MQA, one block per query block (small_gqa_k1)
        (8, 1, 8, 64, 6, True, None, "smallsplit"),  # MQA split-KV
        (4, 1, 16, 16, 16, False, 0.5, "smallcluster"),  # the h1-m1024-k16 row at g = 4
        (4, 1, 16, 64, 18, False, None, "smallsplitcluster"),  # small_gqa_k6sc3
        (4, 2, 160, 16, 12, True, None, "persistent"),  # pair plan: head-pair tiles
        (
            2,
            2,
            33,
            64,
            64,
            True,
            None,
            "persistent",
        ),  # full lists, odd query-block count (lone block)
        (8, 1, 8, 64, 64, False, 0.0, "persistent"),  # MQA, 32 query blocks: split mode
        (4, 4, 64, 64, 32, False, None, None),  # auto: uniform persistent plan
        # a uniform multi-wave split plan (256 mode-1 tiles > 132 SMs): re-paired (mode 4) by the planner of
        # record's ``repair_split`` "auto"; held to the absolute gates, the ratios to the MHA route recorded
        (4, 2, 32, 32, 8, False, None, "persistent"),
    ],
)
def test_gqa_routes_match_the_mha_route_on_replicated_inputs(
    g, h_kv, mb, nb, capacity, ragged, scale, route, record_property
):
    """``num_qo_heads = g x num_kv_heads`` plans the GQA stage of the route the MHA plan of the replicated
    mask takes, on the kv-head-shaped mask and ``k`` / ``v``; its output equals the MHA route's on replicated
    inputs (bit for bit on the static programs), matches the FP32 reference, is written completely into a
    NaN-prefilled buffer and is bit-identical across three launches."""
    h = g * h_kv
    mask = _random_mask(
        h_kv, mb, nb, capacity, seed=h_kv * 1000 + mb, ragged=ragged, device="cuda"
    )
    mask_rep = _replicated(mask, g)
    q, k, v = _gqa_inputs(h, h_kv, mb, nb)
    k_rep, v_rep = _replicated(k, g), _replicated(v, g)
    gqa = CakeVsaSm90Plan(
        "cuda", mask, *_descriptors_of(mask), h, h_kv, 128, sm_scale=scale, route=route
    )
    mha = CakeVsaSm90Plan(
        "cuda",
        mask_rep,
        *_descriptors_of(mask_rep),
        h,
        h,
        128,
        sm_scale=scale,
        route=route,
    )
    assert (gqa.gqa, gqa.g_shift, gqa.num_qo_heads, gqa.num_kv_heads) == (
        g,
        g.bit_length() - 1,
        h,
        h_kv,
    )
    assert (mha.gqa, mha.g_shift, mha.num_kv_heads) == (1, 0, h)
    assert gqa.stage == _gqa_stage_of(mha.stage) and loader.stage_gqa(gqa.stage)
    assert gqa.mode == mha.mode and gqa.num_heads == mha.num_heads == h
    assert not mha.repair_split
    record_property(f"repair_split_g{g}", gqa.repair_split)
    if gqa.small_kmax is None:
        assert gqa.repair_split == (gqa.mode == "split" and not ragged and h * mb > 132)
        assert gqa.num_tiles == _gqa_tile_count(g, h_kv, mb, gqa.mode, gqa.repair_split)
        # header halfword 2 and plan word 2 carry the kv head
        assert int(gqa.hdr[2]) == int(gqa.hdr[0]) // g
        assert int(gqa.meta[0, 2]) == int(gqa.meta[0, 0]) // g
    else:
        assert torch.equal(gqa.plan_param, mha.plan_param)
        assert (gqa.num_tiles, gqa.num_items) == (mha.num_tiles, mha.num_items)
        assert gqa._bound["g_shift"] == gqa.g_shift
        assert "g_shift" not in mha._bound
    out = torch.full(
        (h * mb * 64, 1, 128), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    first = gqa.run(q, k, v, out=out).clone()
    assert first.shape == q.shape and bool(torch.isfinite(first).all()), (
        "an output row was never written or holds NaN"
    )
    for _ in range(2):
        later = gqa.run(q, k, v)
        torch.cuda.synchronize()
        torch.testing.assert_close(later, first, atol=0, rtol=0)
    if gqa.small_split:
        assert int(gqa.counters.abs().sum()) == 0
    expected = mha.run(q, k_rep, v_rep)
    reference = _reference(
        q, k_rep, v_rep, mask_rep, 128**-0.5 if scale is None else scale
    )
    _check_gqa_against_mha(
        gqa, first, expected, reference, record_property, f"g{g}_{gqa.stage}"
    )
    torch.testing.assert_close(first.float(), reference, atol=1e-2, rtol=1e-2)
    assert float((first.float() - reference).abs().max()) <= 0.03
    # k / v are checked against the kv heads, q and out against the query heads
    with pytest.raises(
        ValueError, match=rf"k must have shape \({h_kv}, {nb * 64}, 128\)"
    ):
        gqa.run(q, k_rep, v)
    with pytest.raises(ValueError, match=rf"q must have shape \({h}, {mb * 64}, 128\)"):
        gqa.run(q[:h_kv], k, v)


@requires_hopper
def test_gqa_queue_kernel_matches_the_static_kernel_bit_exact(
    monkeypatch, record_property
):
    """The GQA queue program only moves head-pair tiles between CTAs: bit-identical to the compact and the
    plain static GQA programs on the same plan, which are bit-identical to the static MHA program on
    replicated inputs; the queue words return to zero after every launch and a captured graph replays it."""
    g, h_kv, mb, nb = (
        4,
        2,
        528,
        64,
    )  # 2112 tiles per data type: >= POOL_MIN_TILES per CTA on 132 SMs
    h = g * h_kv
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    mask = _topk_mask(h_kv, mb, nb, 8).cuda()
    mask_rep = _replicated(mask, g)
    q, k, v = _gqa_inputs(h, h_kv, mb, nb)
    queued = CakeVsaSm90Plan(
        "cuda", mask, *_descriptors_of(mask), h, h_kv, 128, route="persistent"
    )
    if not queued.queue:
        pytest.skip(f"{sms} SMs: fewer than {POOL_MIN_TILES} tiles per CTA")
    assert queued.stage == "attention_gqa_queue"
    out = torch.empty((h * mb * 64, 1, 128), device="cuda", dtype=q.dtype)
    first = queued.run(q, k, v, out=out).clone()
    torch.cuda.synchronize()
    assert int(queued.dbg[62:64].abs().sum()) == 0
    # without the queue the planner picks the compact static program for these short head-pair
    # chains (<= 4 ring positions, >= 4 tiles per CTA); the plain static program is forced next
    monkeypatch.setattr(planner, "POOL_MIN_TILES", 10**9)
    compact = CakeVsaSm90Plan(
        "cuda", mask, *_descriptors_of(mask), h, h_kv, 128, route="persistent"
    )
    assert not compact.queue and compact.stage == "attention_gqa_compact"
    assert compact.num_ctas == queued.num_ctas
    torch.testing.assert_close(compact.run(q, k, v), first, atol=0, rtol=0)
    monkeypatch.setattr(planner, "COMPACT_MIN_TILES", 10**9)
    static = CakeVsaSm90Plan(
        "cuda", mask, *_descriptors_of(mask), h, h_kv, 128, route="persistent"
    )
    assert not static.queue and static.stage == "attention_gqa"
    assert static.num_ctas == queued.num_ctas
    torch.testing.assert_close(static.run(q, k, v), first, atol=0, rtol=0)
    mha = CakeVsaSm90Plan(
        "cuda", mask_rep, *_descriptors_of(mask_rep), h, h, 128, route="persistent"
    )
    assert not mha.queue and mha.stage == "attention"
    mha_out = mha.run(q, _replicated(k, g), _replicated(v, g))
    reference = _reference(q, _replicated(k, g), _replicated(v, g), mask_rep, 128**-0.5)
    _check_gqa_against_mha(
        static, first, mha_out, reference, record_property, "queue_vs_static_mha"
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        queued.run(q, k, v, out=out)
    for _ in range(3):
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out.view_as(q), first, atol=0, rtol=0)
        assert int(queued.dbg[62:64].abs().sum()) == 0
    torch.testing.assert_close(first.float(), reference, atol=1e-2, rtol=1e-2)


@requires_hopper
def test_gqa_tagged_kernel_matches_the_untagged_and_mha_programs(
    monkeypatch, record_property
):
    """A g = 2 pair plan whose K / V^T footprint exceeds the L2 budget while its re-read blocks fit it (16 hot
    blocks per kv head, 16 private blocks per query block): the planner tags the single-reader blocks, keyed by
    kv head, and launches the tagged GQA program.  Tags only steer L2 residency: the output is bit-identical
    to the untagged static GQA program on the same plan and to the MHA route on replicated inputs, and matches
    the FP32 reference."""
    g, h_kv, mb, hot, private = 2, 8, 16, 16, 16
    h = g * h_kv
    mask = _hot_and_private_mask(h_kv, mb, hot, private).cuda()
    nb = mask.shape[2]
    mask_rep = _replicated(mask, g)
    q, k, v = _gqa_inputs(h, h_kv, mb, nb)
    k_rep, v_rep = _replicated(k, g), _replicated(v, g)
    tagged = CakeVsaSm90Plan(
        "cuda", mask, *_descriptors_of(mask), h, h_kv, 128, route="persistent"
    )
    assert tagged.mode == "pair" and not tagged.queue and not tagged.compact
    assert tagged.tagged and tagged.stage == "attention_gqa_tagged"
    symbol = loader.MODULES["attention_gqa_tagged"]["kernel_symbol"]
    scratch = torch.empty((h * mb * 64, 1, 128), device="cuda", dtype=q.dtype)
    names = _cuda_kernel_names(lambda: tagged.run(q, k, v, out=scratch))
    assert [name for name in names if name.startswith("kernel_")] == [symbol], names
    record_property("gqa_tagged_stage", tagged.stage)
    record_property("gqa_tagged_kernel", symbol)
    first = tagged.run(q, k, v).clone()
    mha = CakeVsaSm90Plan(
        "cuda", mask_rep, *_descriptors_of(mask_rep), h, h, 128, route="persistent"
    )
    assert mha.stage in ("attention", "attention_tagged")  # a static MHA program
    record_property("mha_stage_on_replicated_inputs", mha.stage)
    reference = _reference(q, k_rep, v_rep, mask_rep, 128**-0.5)
    _check_gqa_against_mha(
        tagged,
        first,
        mha.run(q, k_rep, v_rep),
        reference,
        record_property,
        f"g{g}_{tagged.stage}",
    )
    monkeypatch.setattr(planner, "KV_TAG_SINGLE_READER", False)
    plain = CakeVsaSm90Plan(
        "cuda", mask, *_descriptors_of(mask), h, h_kv, 128, route="persistent"
    )
    assert not plain.tagged and plain.stage == "attention_gqa"
    assert (plain.num_ctas, plain.num_tiles) == (tagged.num_ctas, tagged.num_tiles)
    torch.testing.assert_close(plain.run(q, k, v), first, atol=0, rtol=0)
    torch.testing.assert_close(first.float(), reference, atol=1e-2, rtol=1e-2)
    assert float((first.float() - reference).abs().max()) <= 0.03


@requires_hopper
@pytest.mark.parametrize(
    "route", ["small", "smallsplit", "smallcluster", "smallsplitcluster", "persistent"]
)
def test_gqa_plan_stream_and_graph_lifetime(route):
    """A GQA plan (MQA, g = 4) built on another stream is captured and replayed bit-exactly into a
    NaN-prefilled output, its split counters return to zero, and a captured plan refuses re-planning."""
    g, h_kv = 4, 1
    h = g * h_kv
    mb, nb = 16, 64 if "split" in route else 16
    capacity = {
        "small": 4,
        "persistent": 16,
        "smallsplit": 16,
        "smallcluster": 16,
        "smallsplitcluster": 24,
    }[route]
    mask = _random_mask(h_kv, mb, nb, capacity, seed=17, ragged=False, device="cuda")
    q, k, v = _gqa_inputs(h, h_kv, mb, nb)
    producer = torch.cuda.Stream()
    with torch.cuda.stream(producer):
        plan = CakeVsaSm90Plan(
            "cuda", mask, *_descriptors_of(mask), h, h_kv, 128, route=route
        )
    assert loader.stage_gqa(plan.stage) and plan.gqa == g
    assert (plan.mode == route) if route != "persistent" else plan.small_kmax is None
    expected = plan.run(q, k, v).clone()
    out = torch.empty((h * mb * 64, 1, 128), device="cuda", dtype=q.dtype)
    plan.run(q, k, v, out=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan.run(q, k, v, out=out)
    for _ in range(3):
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out.view_as(expected), expected, atol=0, rtol=0)
        if plan.small_split:
            assert int(plan.counters.abs().sum()) == 0
    with pytest.raises(RuntimeError, match="captured"):
        plan.check_replan()
    reference = _reference(
        q, _replicated(k, g), _replicated(v, g), _replicated(mask, g), 128**-0.5
    )
    torch.testing.assert_close(expected.float(), reference, atol=1e-2, rtol=1e-2)


@requires_hopper
def test_gqa_plan_validates_head_counts_descriptors_and_run_shapes():
    """The plan object: ``g`` in {2, 4, 8} (other multiples and non-multiples refused by name), the mask and
    the descriptors per kv head, an empty query block refused with the MHA message, and ``run`` checking
    ``q`` / ``out`` by the query heads and ``k`` / ``v`` by the kv heads."""
    mask = _random_mask(2, 4, 8, 3, seed=5, ragged=True, device="cuda")
    rows, cols = _descriptors_of(mask)
    for h, kv_h, match in (
        (6, 2, r"g in \(2, 4, 8\)"),
        (10, 2, r"g in \(2, 4, 8\)"),
        (12, 2, r"g in \(2, 4, 8\)"),
        (32, 2, r"g in \(2, 4, 8\)"),
        (5, 2, r"num_qo_heads \(5\) must be a multiple of num_kv_heads \(2\)"),
        (0, 2, "positive"),
        (4, 0, "positive"),
        (8, 4, r"block_mask_map.shape\[0\] \(2\) must equal num_kv_heads \(4\)"),
        (2, 1, r"block_mask_map.shape\[0\] \(2\) must equal num_kv_heads \(1\)"),
    ):
        with pytest.raises(ValueError, match=match):
            CakeVsaSm90Plan("cuda", mask, rows, cols, h, kv_h, 128)
    with pytest.raises(ValueError, match=r"block_row_sz must have shape \(2, 4\)"):
        CakeVsaSm90Plan("cuda", mask, _replicated(rows, 4), cols, 8, 2, 128)
    with pytest.raises(ValueError, match=r"block_col_sz must have shape \(2, 8\)"):
        CakeVsaSm90Plan("cuda", mask, rows, _replicated(cols, 4), 8, 2, 128)
    empty = mask.clone()
    empty[1, 2] = False
    messages = {}
    for h in (2, 8):
        with pytest.raises(
            ValueError, match="empty sparse rows|at least one KV block"
        ) as info:
            CakeVsaSm90Plan("cuda", empty, rows, cols, h, 2, 128)
        messages[h] = str(info.value)
    assert messages[2] == messages[8]
    plan = CakeVsaSm90Plan("cuda", mask, rows, cols, 8, 2, 128)
    assert plan.gqa == 4 and loader.stage_gqa(plan.stage)
    q, k, v = _gqa_inputs(8, 2, 4, 8)
    with pytest.raises(ValueError, match=r"q must have shape \(8, 256, 128\)"):
        plan.run(q[:2], k, v)
    with pytest.raises(ValueError, match=r"k must have shape \(2, 512, 128\)"):
        plan.run(q, _replicated(k, 4), v)
    with pytest.raises(ValueError, match=r"v must have shape \(2, 512, 128\)"):
        plan.run(q, k, _replicated(v, 4))
    with pytest.raises(ValueError, match=r"out must have shape \(2048, 1, 128\)"):
        plan.run(q, k, v, out=torch.empty((512, 1, 128), device="cuda", dtype=q.dtype))
    assert plan.run(q, k, v).shape == (8, 256, 128)


# --------------------------------------------------------------------------- GQA on the Sage FP8 route


def _fp8_gqa_inputs(
    h,
    h_kv,
    mb,
    nb,
    seed=7,
    qscale_mode="unit",
    kscale_mode="unit",
    vscale_mode="unit",
    quantize=None,
):
    """``_fp8_inputs`` for a GQA problem: ``q`` over the query heads, ``k`` / ``v`` over the kv heads (the
    spike lands on one key per kv head), quantized per tensor."""
    q, k, v = (t.clone() for t in _gqa_inputs(h, h_kv, mb, nb, seed))
    if qscale_mode == "x3":
        q.mul_(3.0)
    if kscale_mode == "spike":
        n = k.shape[1]
        for head in range(h_kv):
            key = (n // 2 + 64 * head + 7) % n
            k[head, key] = (k[head, key].float() * 2.0**SPIKE_LOG2).bfloat16()
    if vscale_mode == "dynamic":
        magnitudes = torch.logspace(*V_DYNAMIC_LOG10, 128, device="cuda")
        v = (v.float() * magnitudes).bfloat16()
    return (quantize or _torch_sage_quantize_hnd)(q, k, v)


def _replicated_fp8(k8, v8, scales, g):
    """The FP8 K / V and the K / V scales replicated per group (``k_scale`` along its head dim 1)."""
    q_scale, k_scale, v_scale = scales
    return (
        _replicated(k8, g),
        _replicated(v8, g),
        [q_scale, _replicated(k_scale, g, dim=1), _replicated(v_scale, g)],
    )


def _fp8_gqa_plan_for(mask, h, scales, scale=None, route=None):
    rows, cols = _descriptors_of(mask)
    return CakeVsaSm90Plan(
        "cuda",
        mask,
        rows,
        cols,
        h,
        mask.shape[0],
        128,
        sm_scale=scale,
        q_data_type=FP8,
        q_scale=scales[0],
        k_scale=scales[1],
        v_scale=scales[2],
        route=route,
    )


@requires_hopper
@pytest.mark.parametrize(
    "g,h_kv,mb,nb,capacity,scale,route,ragged",
    [
        (2, 2, 4, 8, 3, None, None, True),  # small
        (4, 1, 4, 4, 1, 0.5, "small", True),  # MQA, small_fp8_gqa_k1
        (8, 1, 8, 64, 6, None, "smallsplit", True),
        (4, 1, 16, 16, 16, -0.125, "smallcluster", True),
        (4, 1, 16, 64, 18, None, "smallsplitcluster", True),
        (4, 2, 160, 16, 12, None, "persistent", True),  # pair plan
        (8, 1, 8, 64, 64, 0.0, "persistent", True),  # MQA, split mode
        # a uniform multi-wave split plan (256 mode-1 tiles > 132 SMs): re-paired (mode 4) by the planner of
        # record's ``repair_split`` "auto"; held to references (a) and (b) at their tolerances, the ratios
        # to the MHA route recorded; reference (a) replays the re-paired plan's own chains
        (4, 2, 32, 32, 8, None, "persistent", False),
    ],
)
def test_fp8_gqa_routes_match_the_fp8_mha_route_on_replicated_inputs(
    g, h_kv, mb, nb, capacity, scale, route, ragged, record_property
):
    """The FP8 GQA stages (``Qs`` per query head, ``Ks`` / ``Vs`` per kv head) on the BF16 GQA planner's
    bytes: three launches bit-identical, both FP8 references on replicated inputs (reference (a) replayed
    from the GQA plan's own tiles), equal to the FP8 MHA route on replicated inputs (bit for bit on the
    static programs; the route gate and the bitwise mismatch count recorded on every program; a re-paired
    plan is held to the two references only)."""
    h = g * h_kv
    mask = _random_mask(
        h_kv, mb, nb, capacity, seed=h_kv * 1000 + mb, ragged=ragged, device="cuda"
    )
    mask_rep = _replicated(mask, g)
    q8, k8, v8, *scales = _fp8_gqa_inputs(h, h_kv, mb, nb)
    k8_rep, v8_rep, scales_rep = _replicated_fp8(k8, v8, scales, g)
    sm_scale = 128**-0.5 if scale is None else scale
    gqa = _fp8_gqa_plan_for(mask, h, scales, scale, route)
    mha = _fp8_plan_for(mask_rep, scales_rep, scale, route)
    assert gqa.fp8 and gqa.gqa == g and loader.stage_dtype(gqa.stage) == "e4m3"
    # the Sage-FP8 route serves static grouped-query plans whose lightest CTA runs at least
    # PIN_MIN_CTA_LOOP_ITERS main-loop iterations on the position-selected static program
    # (the pair plan of this list runs 13 and takes it from the threshold 8 on); its MHA twin
    # is the plain static program, and the bitwise checks below hold on it (pin == static
    # bits).  The expectation comes from the planner decision on the plan's own device, not
    # from the stage the route picked, so a plan routed away from the pin program fails here.
    expected_stage = _gqa_stage_of(mha.stage)
    sms = torch.cuda.get_device_properties(gqa.device).multi_processor_count
    if planner.fp8_gqa_pin(
        plan_vsa_sm90(mask.cpu(), sms=sms, gqa=g, split_tile_order="group")
    ):
        assert expected_stage == "attention_fp8_gqa"
        expected_stage = "attention_fp8_gqa_pin"
    assert gqa.stage == expected_stage and loader.stage_gqa(gqa.stage)
    assert gqa.mode == mha.mode
    record_property(f"fp8_repair_split_g{g}", gqa.repair_split)
    if gqa.small_kmax is None:
        assert gqa.repair_split == (gqa.mode == "split" and not ragged and h * mb > 132)
    assert gqa.k_scale_groups == mha.k_scale_groups == -(-(nb * 64) // 16)
    # the FP8 GQA plan consumes exactly the BF16 GQA plan's bytes
    bf16 = CakeVsaSm90Plan(
        "cuda", mask, *_descriptors_of(mask), h, h_kv, 128, sm_scale=scale, route=route
    )
    if gqa.small_kmax is None:
        assert torch.equal(gqa.meta, bf16.meta) and torch.equal(gqa.hdr, bf16.hdr)
    else:
        assert torch.equal(gqa.plan_param, bf16.plan_param)
    outs = [gqa.run(q8, k8, v8).clone() for _ in range(3)]
    assert outs[0].dtype == torch.bfloat16 and outs[0].shape == (h, mb * 64, 128)
    for later in outs[1:]:
        torch.testing.assert_close(later, outs[0], atol=0, rtol=0)
    exact = _fp8_reference_exact(q8, k8_rep, v8_rep, scales_rep, mask_rep, sm_scale)
    _check_gqa_against_mha(
        gqa,
        outs[0],
        mha.run(q8, k8_rep, v8_rep),
        exact,
        record_property,
        f"fp8_g{g}_{gqa.stage}",
    )
    # (a) the plan-faithful replay of the GQA plan's own tiles (the MHA tiles' chains unless re-paired)
    # and (b) the exact softmax of the dequantized inputs
    _fp8_check(
        gqa, outs[0], q8, k8_rep, v8_rep, scales_rep, mask_rep, sm_scale, mask_kv=mask
    )


@requires_hopper
def test_fp8_gqa_queue_route_takes_the_shared_transposer_for_head_pair_plans():
    """The Sage-FP8 route runs grouped-query queue plans of head-pair tiles on the
    shared-transposer queue program (``attention_fp8_gqa_xs_queue``); split-mode queue
    plans keep ``attention_fp8_gqa_queue``, the static grouped-query plans and the MHA
    plans their programs (``planner.fp8_gqa_xshare`` and ``loader.persistent_stage``;
    host rules, no GPU).  The split-tile queue launch is the mode-1 replay
    (``repair_split=False``); under the planner default ``"auto"`` the same uniform
    multi-wave shape is re-paired, and that plan never takes the shared transposer
    either."""
    sms = 132
    g = 4
    pair = planner.plan_vsa_sm90(_topk_mask(2, 528, 64, 8), sms=sms, gqa=g)
    assert (pair["mode"], bool(pair["queue"])) == ("pair", True)
    assert planner.fp8_gqa_xshare(pair) is True
    assert (
        loader.persistent_stage(True, fp8=True, gqa=True, xshare=True)
        == "attention_fp8_gqa_xs_queue"
    )
    split_mask = _topk_mask(2, 128, 128, 2)
    split = planner.plan_vsa_sm90(split_mask, sms=sms, gqa=g, repair_split=False)
    assert (split["mode"], bool(split["queue"]), split["repair_split"]) == (
        "split",
        True,
        False,
    )
    assert planner.fp8_gqa_xshare(split) is False
    repaired = planner.plan_vsa_sm90(split_mask, sms=sms, gqa=g)
    assert (repaired["mode"], repaired["repair_split"]) == ("split", True)
    assert planner.fp8_gqa_xshare(repaired) is False
    static = planner.plan_vsa_sm90(_topk_mask(2, 16, 16, 8), sms=sms, gqa=g)
    assert not static["queue"] and planner.fp8_gqa_xshare(static) is False
    mha = planner.plan_vsa_sm90(_replicated(_topk_mask(2, 528, 64, 8), g), sms=sms)
    assert mha["queue"] and mha["gqa"] == 1 and planner.fp8_gqa_xshare(mha) is False
    for plan in (pair, split, repaired, static, mha):
        stage = loader.persistent_stage(
            bool(plan["queue"]),
            bool(plan["compact"]),
            bool(plan["tagged"]),
            fp8=True,
            gqa=plan["gqa"] > 1,
            xshare=planner.fp8_gqa_xshare(plan),
        )
        assert stage.endswith("_xs_queue") is (plan is pair), stage
        assert loader.persistent_stage(
            bool(plan["queue"]),
            bool(plan["compact"]),
            bool(plan["tagged"]),
            gqa=plan["gqa"] > 1,
        ).startswith("attention")
    # the rejected forms: no static form, no BF16 twin, no MHA twin
    with pytest.raises(ValueError, match="shared-transposer"):
        loader.persistent_stage(False, fp8=True, gqa=True, xshare=True)
    with pytest.raises(ValueError, match="shared-transposer"):
        loader.persistent_stage(True, fp8=False, gqa=True, xshare=True)
    with pytest.raises(ValueError, match="shared-transposer"):
        loader.persistent_stage(True, fp8=True, gqa=False, xshare=True)
    with pytest.raises(ValueError, match="shared-transposer"):
        loader.persistent_stage(True, compact=True, fp8=True, gqa=True, xshare=True)


def test_fp8_gqa_pin_route_takes_long_static_plans_only():
    """The Sage-FP8 route runs static grouped-query plans whose lightest CTA runs at least
    ``PIN_MIN_CTA_LOOP_ITERS`` main-loop iterations on the position-selected static program
    (``attention_fp8_gqa_pin``); shorter static plans keep ``attention_fp8_gqa``, the queue, compact
    and tagged grouped-query plans and the MHA plans their programs (``planner.fp8_gqa_pin`` and
    ``loader.persistent_stage``; host rules, no GPU)."""
    sms = 132
    g = 4
    long = planner.plan_vsa_sm90(_topk_mask(2, 64, 64, 32), sms=sms, gqa=g)
    assert (long["mode"], bool(long["queue"]), long["pin"]) == ("pair", False, True)
    assert planner.fp8_gqa_pin(long) is True
    assert (
        loader.persistent_stage(False, fp8=True, gqa=True, pin=True)
        == "attention_fp8_gqa_pin"
    )
    short = planner.plan_vsa_sm90(_topk_mask(2, 32, 32, 16), sms=sms, gqa=g)
    assert not short["queue"] and short["pin"] is False
    assert planner.fp8_gqa_pin(short) is False
    compact = planner.plan_vsa_sm90(_topk_mask(4, 128, 128, 5), sms=sms, gqa=g)
    assert compact["compact"] and compact["pin"] is False
    queue = planner.plan_vsa_sm90(_topk_mask(2, 528, 64, 8), sms=sms, gqa=g)
    assert queue["queue"] and queue["pin"] is False
    mha = planner.plan_vsa_sm90(_replicated(_topk_mask(2, 64, 64, 32), g), sms=sms)
    assert mha["gqa"] == 1 and mha["pin"] is False and planner.fp8_gqa_pin(mha) is False
    for plan in (long, short, compact, queue, mha):
        stage = loader.persistent_stage(
            bool(plan["queue"]),
            bool(plan["compact"]),
            bool(plan["tagged"]),
            fp8=True,
            gqa=plan["gqa"] > 1,
            xshare=planner.fp8_gqa_xshare(plan),
            pin=planner.fp8_gqa_pin(plan),
        )
        assert stage.endswith("_pin") is (plan is long), stage
        assert loader.persistent_stage(
            bool(plan["queue"]),
            bool(plan["compact"]),
            bool(plan["tagged"]),
            gqa=plan["gqa"] > 1,
        ).startswith("attention")
    assert loader.persistent_stage(False, fp8=True, gqa=True) == "attention_fp8_gqa"
    # the rejected forms: the FP8 grouped-query static program only
    for kwargs in (
        dict(queue=True, fp8=True, gqa=True, pin=True),
        dict(queue=False, compact=True, fp8=True, gqa=True, pin=True),
        dict(queue=False, tagged=True, fp8=True, gqa=True, pin=True),
        dict(queue=False, fp8=False, gqa=True, pin=True),
        dict(queue=False, fp8=True, gqa=False, pin=True),
        dict(queue=True, fp8=True, gqa=True, xshare=True, pin=True),
    ):
        with pytest.raises(ValueError, match="position-selected"):
            loader.persistent_stage(**kwargs)


def test_fp8_gqa_queue_kernel_matches_the_static_kernel_bit_exact(
    monkeypatch, record_property
):
    """The FP8 GQA shared-transposer queue program (the program of head-pair queue plans) only moves
    head-pair tiles between CTAs and the V^T transposition between producer warps: bit-identical to the
    compact, the position-selected and the plain static FP8 GQA programs on the same plan, which are
    bit-identical to the static FP8 MHA program on replicated inputs; the queue words return to zero
    after every launch and a captured graph replays it."""
    g, h_kv, mb, nb = (
        4,
        2,
        528,
        64,
    )  # 2112 head-pair tiles: >= POOL_MIN_TILES per CTA on 132 SMs
    h = g * h_kv
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    mask = _topk_mask(h_kv, mb, nb, 8).cuda()
    mask_rep = _replicated(mask, g)
    q8, k8, v8, *scales = _fp8_gqa_inputs(h, h_kv, mb, nb)
    k8_rep, v8_rep, scales_rep = _replicated_fp8(k8, v8, scales, g)
    queued = _fp8_gqa_plan_for(mask, h, scales, route="persistent")
    if not queued.queue:
        pytest.skip(f"{sms} SMs: fewer than {POOL_MIN_TILES} tiles per CTA")
    assert queued.stage == "attention_fp8_gqa_xs_queue"
    record_property("fp8_gqa_queue_stage", queued.stage)
    out = torch.empty((h * mb * 64, 1, 128), device="cuda", dtype=torch.bfloat16)
    first = queued.run(q8, k8, v8, out=out).clone()
    torch.cuda.synchronize()
    assert int(queued.dbg[62:64].abs().sum()) == 0
    # without the queue the planner picks the compact static program for these short head-pair
    # chains (<= 4 ring positions, >= 4 tiles per CTA); the plain static program is forced next
    monkeypatch.setattr(planner, "POOL_MIN_TILES", 10**9)
    compact = _fp8_gqa_plan_for(mask, h, scales, route="persistent")
    assert not compact.queue and compact.stage == "attention_fp8_gqa_compact"
    assert compact.num_ctas == queued.num_ctas
    torch.testing.assert_close(compact.run(q8, k8, v8), first, atol=0, rtol=0)
    # without the compact program the planner takes the position-selected static program for these
    # 16 tiles per CTA (cta_loop_iters 55 at 132 SMs in the round-16 dry-run, >= PIN_MIN_CTA_LOOP_ITERS);
    # the plain static program is forced last by raising that threshold past the statistic
    monkeypatch.setattr(planner, "COMPACT_MIN_TILES", 10**9)
    pinned = _fp8_gqa_plan_for(mask, h, scales, route="persistent")
    assert not pinned.queue and pinned.stage == "attention_fp8_gqa_pin"
    assert pinned.num_ctas == queued.num_ctas
    record_property("fp8_gqa_pin_stage_on_the_queue_plan", pinned.stage)
    torch.testing.assert_close(pinned.run(q8, k8, v8), first, atol=0, rtol=0)
    monkeypatch.setattr(planner, "PIN_MIN_CTA_LOOP_ITERS", 10**9)
    static = _fp8_gqa_plan_for(mask, h, scales, route="persistent")
    assert not static.queue and static.stage == "attention_fp8_gqa"
    assert static.num_ctas == queued.num_ctas
    torch.testing.assert_close(static.run(q8, k8, v8), first, atol=0, rtol=0)
    mha = _fp8_plan_for(mask_rep, scales_rep, route="persistent")
    assert not mha.queue and mha.stage == "attention_fp8"
    exact = _fp8_reference_exact(q8, k8_rep, v8_rep, scales_rep, mask_rep, 128**-0.5)
    _check_gqa_against_mha(
        static,
        first,
        mha.run(q8, k8_rep, v8_rep),
        exact,
        record_property,
        "fp8_queue_vs_static_mha",
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        queued.run(q8, k8, v8, out=out)
    for _ in range(3):
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out.view_as(first), first, atol=0, rtol=0)
        assert int(queued.dbg[62:64].abs().sum()) == 0
    torch.testing.assert_close(first.float(), exact, atol=5e-2, rtol=5e-2)


@requires_hopper
def test_fp8_gqa_tagged_kernel_matches_the_untagged_and_mha_programs(
    monkeypatch, record_property
):
    """The FP8 twin of the tagged GQA case on the same plan bytes: the tagged FP8 GQA program launches, its
    output is bit-identical to the untagged static FP8 GQA program and to the FP8 MHA route on replicated
    inputs, and both FP8 references hold."""
    g, h_kv, mb, hot, private = 2, 8, 16, 16, 16
    h = g * h_kv
    mask = _hot_and_private_mask(h_kv, mb, hot, private).cuda()
    nb = mask.shape[2]
    mask_rep = _replicated(mask, g)
    q8, k8, v8, *scales = _fp8_gqa_inputs(h, h_kv, mb, nb)
    k8_rep, v8_rep, scales_rep = _replicated_fp8(k8, v8, scales, g)
    tagged = _fp8_gqa_plan_for(mask, h, scales, route="persistent")
    assert tagged.mode == "pair" and not tagged.queue and not tagged.compact
    assert tagged.tagged and tagged.stage == "attention_fp8_gqa_tagged"
    symbol = loader.MODULES["attention_fp8_gqa_tagged"]["kernel_symbol"]
    scratch = torch.empty((h * mb * 64, 1, 128), device="cuda", dtype=torch.bfloat16)
    names = _cuda_kernel_names(lambda: tagged.run(q8, k8, v8, out=scratch))
    assert [name for name in names if name.startswith("kernel_")] == [symbol], names
    record_property("fp8_gqa_tagged_stage", tagged.stage)
    record_property("fp8_gqa_tagged_kernel", symbol)
    first = tagged.run(q8, k8, v8).clone()
    mha = _fp8_plan_for(mask_rep, scales_rep, route="persistent")
    assert mha.stage in ("attention_fp8", "attention_fp8_tagged")
    record_property("fp8_mha_stage_on_replicated_inputs", mha.stage)
    exact = _fp8_reference_exact(q8, k8_rep, v8_rep, scales_rep, mask_rep, 128**-0.5)
    _check_gqa_against_mha(
        tagged,
        first,
        mha.run(q8, k8_rep, v8_rep),
        exact,
        record_property,
        f"fp8_g{g}_{tagged.stage}",
    )
    _fp8_check(tagged, first, q8, k8_rep, v8_rep, scales_rep, mask_rep, 128**-0.5)
    # the untagged plan runs 15 main-loop iterations per CTA (one 16-position head-pair tile):
    # at the shipped PIN_MIN_CTA_LOOP_ITERS it takes the position-selected static program, which
    # is checked bit-exact first; the plain static program is forced next by raising that
    # threshold past the statistic
    monkeypatch.setattr(planner, "KV_TAG_SINGLE_READER", False)
    pinned = _fp8_gqa_plan_for(mask, h, scales, route="persistent")
    assert not pinned.tagged and pinned.stage == "attention_fp8_gqa_pin"
    assert (pinned.num_ctas, pinned.num_tiles) == (tagged.num_ctas, tagged.num_tiles)
    record_property("fp8_gqa_pin_stage_on_the_untagged_plan", pinned.stage)
    torch.testing.assert_close(pinned.run(q8, k8, v8), first, atol=0, rtol=0)
    monkeypatch.setattr(planner, "PIN_MIN_CTA_LOOP_ITERS", 10**9)
    plain = _fp8_gqa_plan_for(mask, h, scales, route="persistent")
    assert not plain.tagged and plain.stage == "attention_fp8_gqa"
    torch.testing.assert_close(plain.run(q8, k8, v8), first, atol=0, rtol=0)


@requires_hopper
def test_fp8_gqa_pin_kernel_matches_the_static_and_mha_programs(
    monkeypatch, record_property
):
    """The position-selected FP8 GQA static program (the program of static grouped-query plans whose
    lightest CTA runs at least PIN_MIN_CTA_LOOP_ITERS main-loop iterations) on a routed shape,
    h8-m4096-n4096-k32 at g = 4 (256 head-pair tiles of 16 positions on 128 CTAs): it launches, its
    output is bit-identical to the plain static FP8 GQA program on the same plan bytes and to the FP8
    MHA route on replicated inputs, both FP8 references hold, and a captured graph replays it."""
    g, h_kv, mb, nb, k = 4, 2, 64, 64, 32
    h = g * h_kv
    mask = _topk_mask(h_kv, mb, nb, k).cuda()
    mask_rep = _replicated(mask, g)
    q8, k8, v8, *scales = _fp8_gqa_inputs(h, h_kv, mb, nb)
    k8_rep, v8_rep, scales_rep = _replicated_fp8(k8, v8, scales, g)
    pinned = _fp8_gqa_plan_for(mask, h, scales, route="persistent")
    if pinned.stage != "attention_fp8_gqa_pin":
        sms = torch.cuda.get_device_properties(0).multi_processor_count
        pytest.skip(
            f"{pinned.stage} on {sms} SMs: the shape is not position-selected here"
        )
    assert pinned.mode == "pair" and not pinned.queue and not pinned.compact
    assert not pinned.tagged and not pinned.repair_split
    symbol = loader.MODULES["attention_fp8_gqa_pin"]["kernel_symbol"]
    out = torch.empty((h * mb * 64, 1, 128), device="cuda", dtype=torch.bfloat16)
    names = _cuda_kernel_names(lambda: pinned.run(q8, k8, v8, out=out))
    assert [name for name in names if name.startswith("kernel_")] == [symbol], names
    record_property("fp8_gqa_pin_stage", pinned.stage)
    record_property("fp8_gqa_pin_kernel", symbol)
    first = pinned.run(q8, k8, v8, out=out).clone()
    torch.cuda.synchronize()
    # the plain static program on the same plan bytes: the threshold raised past the statistic
    monkeypatch.setattr(planner, "PIN_MIN_CTA_LOOP_ITERS", 10**9)
    static = _fp8_gqa_plan_for(mask, h, scales, route="persistent")
    assert not static.queue and static.stage == "attention_fp8_gqa"
    assert static.num_ctas == pinned.num_ctas
    assert torch.equal(static.meta, pinned.meta) and torch.equal(static.hdr, pinned.hdr)
    torch.testing.assert_close(static.run(q8, k8, v8), first, atol=0, rtol=0)
    mha = _fp8_plan_for(mask_rep, scales_rep, route="persistent")
    assert mha.stage in ("attention_fp8", "attention_fp8_tagged")
    record_property("fp8_mha_stage_on_replicated_inputs", mha.stage)
    exact = _fp8_reference_exact(q8, k8_rep, v8_rep, scales_rep, mask_rep, 128**-0.5)
    _check_gqa_against_mha(
        pinned,
        first,
        mha.run(q8, k8_rep, v8_rep),
        exact,
        record_property,
        "fp8_pin_vs_static_mha",
    )
    _fp8_check(pinned, first, q8, k8_rep, v8_rep, scales_rep, mask_rep, 128**-0.5)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pinned.run(q8, k8, v8, out=out)
    for _ in range(3):
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out.view_as(first), first, atol=0, rtol=0)


@requires_hopper
@pytest.mark.parametrize("variant", ["x3", "spike", "dynamic", "zero_scale"])
def test_fp8_gqa_scale_variants(variant):
    """Scale addressing by query head (``q_scale``) versus kv head (``k_scale`` / ``v_scale``) on a ragged
    pair plan at g = 4 with two kv heads: the scale variants of the FP8 MHA tests, both references, bit for
    bit the MHA route on replicated inputs."""
    g, h_kv, mb, nb = 4, 2, 16, 16
    h = g * h_kv
    mask = _random_mask(h_kv, mb, nb, 8, seed=23, ragged=True, device="cuda")
    mask_rep = _replicated(mask, g)
    modes = {
        "x3": {"qscale_mode": "x3"},
        "spike": {"kscale_mode": "spike"},
        "dynamic": {"vscale_mode": "dynamic"},
        "zero_scale": {},
    }[variant]
    scale = 0.0 if variant == "zero_scale" else None
    q8, k8, v8, *scales = _fp8_gqa_inputs(h, h_kv, mb, nb, **modes)
    k8_rep, v8_rep, scales_rep = _replicated_fp8(k8, v8, scales, g)
    sm_scale = 128**-0.5 if scale is None else scale
    for route in ("persistent", None):
        plan = _fp8_gqa_plan_for(mask, h, scales, scale, route=route)
        out = plan.run(q8, k8, v8)
        _fp8_check(plan, out, q8, k8_rep, v8_rep, scales_rep, mask_rep, sm_scale)
        mha = _fp8_plan_for(mask_rep, scales_rep, scale, route=route)
        assert plan.stage == _gqa_stage_of(mha.stage)
        if _static_route(plan):
            torch.testing.assert_close(out, mha.run(q8, k8_rep, v8_rep), atol=0, rtol=0)


@requires_hopper
@pytest.mark.parametrize("route", ["small", "persistent"])
def test_fp8_gqa_plan_stream_and_graph_lifetime(route):
    """An FP8 GQA plan built on another stream is captured and replayed bit-exactly into a NaN-prefilled
    output; re-quantizing new inputs into the captured tensors (the package quantizer on GQA shapes) is what
    the replay computes."""
    g, h_kv, mb, nb = 4, 1, 16, 16
    h = g * h_kv
    capacity = {"small": 4, "persistent": 16}[route]
    mask = _random_mask(h_kv, mb, nb, capacity, seed=17, ragged=False, device="cuda")
    q, k, v = (t.clone() for t in _gqa_inputs(h, h_kv, mb, nb))
    producer = torch.cuda.Stream()
    with torch.cuda.stream(producer):
        q8, k8, v8, q_scale, k_scale, v_scale = pkg.sage_fp8_quantize_sm90(q, k, v)
        plan = _fp8_gqa_plan_for(mask, h, (q_scale, k_scale, v_scale), route=route)
    assert plan.fp8 and plan.gqa == g and loader.stage_gqa(plan.stage)
    expected = plan.run(q8, k8, v8).clone()
    out = torch.empty((h * mb * 64, 1, 128), device="cuda", dtype=torch.bfloat16)
    plan.run(q8, k8, v8, out=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan.run(q8, k8, v8, out=out)
    for _ in range(3):
        out.fill_(float("nan"))
        graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out.view_as(expected), expected, atol=0, rtol=0)
    q.normal_()
    v.mul_(3.0)
    new = pkg.sage_fp8_quantize_sm90(q, k, v)
    for captured, fresh in zip(
        (q8, k8, v8, q_scale, k_scale, v_scale), new, strict=True
    ):
        captured.copy_(fresh)
    graph.replay()
    torch.cuda.synchronize()
    k8_rep, v8_rep, scales_rep = _replicated_fp8(k8, v8, (q_scale, k_scale, v_scale), g)
    exact = _fp8_reference_exact(
        q8, k8_rep, v8_rep, scales_rep, _replicated(mask, g), 128**-0.5
    )
    torch.testing.assert_close(
        out.view_as(expected).float(), exact, atol=5e-2, rtol=5e-2
    )
    with pytest.raises(RuntimeError, match="captured"):
        plan.check_replan()


@requires_hopper
@pytest.mark.parametrize(
    "shape",
    [(2, 1, 256, 256), (8, 2, 1000, 1500), (4, 1, 65, 4000), (16, 2, 4096, 4096)],
)
def test_sage_fp8_quantize_sm90_gqa_matches_torch(shape, record_property):
    """The package quantizer on GQA shapes (``k`` / ``v`` over the kv heads): the SM100 criterion against the
    torch recipe applied per tensor, the plan's scale shapes, the same bits as the MHA path on replicated
    inputs, and exactly three kernels (the qk / V-amax kernel over the query heads, then over the kv heads,
    then the V kernel) behind one fill."""
    h, h_kv, sq, sk = shape
    g = h // h_kv
    gen = torch.Generator(device="cuda").manual_seed(sq * 7 + sk)
    q = torch.randn((h, sq, 128), device="cuda", generator=gen).bfloat16()
    k = torch.randn((h_kv, sk, 128), device="cuda", generator=gen).bfloat16()
    v = torch.randn((h_kv, sk, 128), device="cuda", generator=gen).bfloat16()
    got = pkg.sage_fp8_quantize_sm90(q, k, v)
    torch.cuda.synchronize()
    ref = _torch_sage_quantize_hnd(q, k, v)
    groups = -(-sk // 16)
    assert [tuple(t.shape) for t in got[3:]] == [
        (1, h, sq),
        (1, h_kv, groups),
        (h_kv, 128),
    ]
    assert all(t.dtype == torch.float32 and t.is_contiguous() for t in got[3:])
    for name, a, b in zip(
        ("q_scale", "k_scale", "v_scale"), got[3:], ref[3:], strict=True
    ):
        assert a.shape == b.shape, name
        rel = float((a - b).abs().max() / b.abs().max())
        assert rel < 1e-5, f"{name}: max relative scale error {rel:.3e}"
        record_property(f"gqa_bitwise_mismatch_{name}", int((a != b).sum()))
    for name, a, b in zip(("q8", "k8", "v8"), got[:3], ref[:3], strict=True):
        assert a.dtype == FP8 and a.shape == b.shape, name
        af, bf = a.float(), b.float()
        step = torch.maximum(af.abs(), bf.abs()).clamp_min(1.0) * 0.125
        mismatch = int(((af - bf).abs() > step).sum())
        assert mismatch == 0, (
            f"{name}: {mismatch} elements differ by more than one E4M3 step"
        )
    # the MHA path on replicated k / v yields the same bits per head
    rep = pkg.sage_fp8_quantize_sm90(q, _replicated(k, g), _replicated(v, g))
    torch.cuda.synchronize()
    assert torch.equal(rep[0].view(torch.uint8), got[0].view(torch.uint8))
    assert torch.equal(
        rep[1].view(torch.uint8), _replicated(got[1], g).view(torch.uint8)
    )
    assert torch.equal(
        rep[2].view(torch.uint8), _replicated(got[2], g).view(torch.uint8)
    )
    assert torch.equal(rep[3], got[3])
    assert torch.equal(rep[4], _replicated(got[4], g, dim=1))
    assert torch.equal(rep[5], _replicated(got[5], g))
    names = _cuda_kernel_names(lambda: pkg.sage_fp8_quantize_sm90(q, k, v))
    kernels = [name for name in names if name.startswith("kernel_")]
    qk = loader.MODULES["quantize_qk_vamax"]["kernel_symbol"]
    assert kernels == [qk, qk, loader.MODULES["quantize_v"]["kernel_symbol"]], names
    assert len(names) == len(kernels) + 1 and "Fill" in names[0], names
