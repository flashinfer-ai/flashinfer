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

"""Cake SM90 VSA (``backend="cake"`` on Hopper): planning, numerics, lifetime, API."""


import warnings

import pytest
import torch

from flashinfer.cake_vsa_sm90 import (
    MAX_NSPLIT,
    PLAN_HALFWORDS,
    CLUSTER_GPC_SMS,
    CLUSTER_VARIANTS,
    PLAN_META_SPLIT,
    PLAN_META_UNSPLIT,
    SMALL_KMAX_VARIANTS,
    SMALL_OCCUPANCY,
    CakeVsaSm90Plan,
    cluster_cost,
    cluster_variant_for,
    split_cost,
    MAX_OWN,
    META_NOWN,
    META_NSEQ,
    META_NTILES,
    META_OWN_OFF,
    META_SEQ_OFF,
    META_WORDS,
    OWN_WORDS,
    _schedule_feasible,
    plan_small,
    plan_vsa_sm90,
    small_route,
    split_kmax,
)


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


# ---------------------------------------------------------------------------
# Host planner (CPU only)
# ---------------------------------------------------------------------------


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
            None,
        ),  # 16 tiles x 64 blocks: no one-wave variant fits the plan budget
    ],
)
def test_small_route_rule(h, mb, nb, capacity, expected):
    mask = _random_mask(h, mb, nb, capacity, ragged=False)
    route = small_route(mask, sms=132)
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
        assert h * mb <= (132 // CLUSTER_GPC_SMS) * (CLUSTER_GPC_SMS // csize)
        assert (kmax, csize) == cluster_variant_for(counts, sms=132)
        assert cluster_cost(counts, kmax, csize) < split_cost(
            counts, split_kmax(counts, sms=132)
        )
    else:
        assert route == expected
    if expected not in (None, "split", "cluster"):
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
    kmax, split, cluster = small_route(ragged, sms=132)
    assert (kmax, split, cluster) == (4, False, 2)
    assert split_kmax(counts, sms=132) in (
        3,
        4,
        6,
    )  # a split variant fits; it is just costlier
    assert cluster_variant_for(counts, sms=132) == (4, 2)
    uniform = torch.zeros((1, 16, 16), dtype=torch.bool)
    uniform[..., :16] = True
    assert small_route(uniform, sms=132) == (4, False, 4)


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
    with pytest.raises(ValueError, match="exclusive"):
        plan_small(
            torch.ones((1, 16, 16), dtype=torch.bool), kmax=4, split=True, cluster=4
        )


def _decode_small_rows(plan):
    stride, meta_halfwords = plan["stride"], plan["meta_halfwords"]
    assert stride == plan["kmax"] + meta_halfwords
    rows = plan["plan"][: plan["num_items"] * stride].view(-1, stride).int()
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


@pytest.mark.parametrize("kmax", [1, 3, 4, 6])
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
        plan_small(torch.ones((1, 1, 64), dtype=torch.bool), kmax=1, split=True)


def _decode_tiles(plan):
    """Yield (head, qb0, qb1, mode, positions[(blk_a, blk_b)], owns[2]) per tile."""
    meta = plan["meta"].numpy().view("uint32")
    stride = plan["tile_stride"]
    for c in range(plan["num_ctas"]):
        n_tiles = int(meta[c * stride, META_NTILES])
        assert 1 <= n_tiles <= stride
        for i in range(n_tiles):
            row = meta[c * stride + i]
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
    assert plan["num_ctas"] == min(plan["num_tiles"], 132)
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


def _cta_tiles(plan):
    """Per-CTA lists of (head, positions) read back from the plan rows."""
    meta, stride = plan["meta"], plan["tile_stride"]
    out = []
    for c in range(plan["num_ctas"]):
        n = int(meta[c * stride, META_NTILES])
        out.append(
            [
                (int(meta[c * stride + i, 0]), int(meta[c * stride + i, META_NSEQ]))
                for i in range(n)
            ]
        )
    return out


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
    for tiles in _cta_tiles(plan):
        heads = [h for h, _ in tiles]
        assert heads == sorted(heads)


def test_plan_mode_selection_prefers_split_below_full_occupancy():
    mask = _random_mask(8, 16, 16, 12, ragged=False)
    assert plan_vsa_sm90(mask, sms=132)["mode"] == "split"
    dense = _random_mask(8, 64, 64, 32, ragged=False)
    assert plan_vsa_sm90(dense, sms=132)["mode"] == "pair"


# ---------------------------------------------------------------------------
# GPU coverage (Hopper only)
# ---------------------------------------------------------------------------

requires_hopper = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="Requires an SM90 GPU",
)


def _wrapper(mask, backend="cake", scale=None):
    from flashinfer.sparse import VariableBlockSparseAttentionWrapper

    h, mb, nb = mask.shape
    workspace_size = 0 if backend == "cake" else 128 * 1024 * 1024
    wrapper = VariableBlockSparseAttentionWrapper(
        torch.empty(workspace_size, device="cuda", dtype=torch.uint8), backend=backend
    )
    wrapper.plan(
        mask,
        torch.full((h, mb), 64, dtype=torch.int32, device=mask.device),
        torch.full((h, nb), 64, dtype=torch.int32, device=mask.device),
        h,
        h,
        128,
        q_data_type=torch.bfloat16,
        sm_scale=scale,
        non_blocking=False,
    )
    return wrapper


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
        (1, 1, 2, 2, 1e-7, False),  # two slices of one block
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
        mask, sms=torch.cuda.get_device_properties(0).multi_processor_count
    )
    expected_mode = _mode_of(rule)
    assert auto.mode == expected_mode


def _mode_of(rule):
    if rule is None:
        return "persistent"
    if rule[2]:
        return "smallcluster"
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
        mask, sms=torch.cuda.get_device_properties(0).multi_processor_count
    )
    assert auto.mode == _mode_of(rule)


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


@requires_hopper
@pytest.mark.parametrize(
    "h,mb,nb,capacity",
    [(2, 1, 2, 1), (1, 16, 6, 3), (1, 128, 8, 4), (1, 128, 16, 12), (2, 160, 16, 12)],
)
def test_all_schedules_against_fa3(h, mb, nb, capacity):
    torch.manual_seed(42)
    mask = torch.zeros((h, mb, nb), dtype=torch.bool, device="cuda")
    for head in range(h):
        for row in range(mb):
            count = capacity if row == 0 else 1 + row % capacity
            mask[head, row, torch.randperm(nb, device="cuda")[:count]] = True
    candidate = _wrapper(mask)
    reference = _wrapper(mask, "fa3")
    q, k, v = _inputs(h, mb, nb)
    out = torch.empty((h * mb * 64, 1, 128), device="cuda", dtype=torch.bfloat16)
    for _ in range(2):
        expected = reference.run(q, k, v, enable_pdl=False)
        actual = candidate.run(q, k, v, out=out, enable_pdl=False)
        assert actual.data_ptr() == out.data_ptr()
        assert actual.shape == q.shape
        torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)
        assert float((actual.float() - expected.float()).abs().max()) <= 0.03
        q.normal_()
        v.normal_()


@requires_hopper
@pytest.mark.parametrize(
    "h,mb,nb,capacity,ragged,scale",
    [
        # boundary / tail: one block, odd block counts, full 64-block capacity,
        # unequal Q/KV lengths, lone tile of a pair plan, custom scale
        (1, 1, 1, 1, False, None),
        (2, 3, 5, 2, True, None),
        (2, 3, 3, 3, False, 0.5),
        (1, 1, 64, 64, False, None),
        (2, 33, 64, 64, True, None),
        (3, 7, 64, 64, False, None),
        (4, 16, 8, 4, True, None),
        (5, 17, 9, 5, True, None),
        (8, 64, 64, 32, False, None),
        (7, 64, 64, 16, True, 0.125),
        (2, 64, 64, 3, True, None),
    ],
)
def test_held_out_shapes_against_fp32_reference(h, mb, nb, capacity, ragged, scale):
    mask = _random_mask(
        h, mb, nb, capacity, seed=h * 1000 + mb, ragged=ragged, device="cuda"
    )
    torch.manual_seed(h * 31 + mb)
    q, k, v = _inputs(h, mb, nb)
    wrapper = _wrapper(mask, scale=scale)
    actual = wrapper.run(q, k, v)
    expected = _reference(q, k, v, mask, 128**-0.5 if scale is None else scale)
    assert bool(torch.isfinite(actual).all())
    torch.testing.assert_close(actual.float(), expected, atol=0.01, rtol=0.01)
    assert float((actual.float() - expected.float()).abs().max()) <= 0.03


@requires_hopper
def test_replan_and_independent_wrappers():
    mask, rows, cols = _descriptors(2, 2, 3, device="cuda")
    mask[:, :, 1:] = False
    first = _wrapper(mask)
    q, k, v = _inputs(2, 2, 3)
    expected_first = first.run(q, k, v).clone()
    mask[:, :, 0] = False
    mask[:, :, 2] = True
    second = _wrapper(mask)
    expected_second = second.run(q, k, v).clone()
    torch.testing.assert_close(first.run(q, k, v), expected_first, atol=0, rtol=0)
    first.plan(mask, rows, cols, 2, 2, 128, q_data_type=torch.bfloat16)
    torch.testing.assert_close(first.run(q, k, v), expected_second, atol=0, rtol=0)
    torch.testing.assert_close(second.run(q, k, v), expected_second, atol=0, rtol=0)


@requires_hopper
@pytest.mark.parametrize("scale", [0.0, -0.125, 1e-7, 128**-0.5])
def test_padding_uses_attention_math(scale):
    mask, _, _ = _descriptors(1, 1, 2, device="cuda")
    mask[:, :, 1] = False
    q, k, v = _inputs(1, 1, 2)
    q.fill_(-32)
    k.fill_(32)
    actual = _wrapper(mask, scale=scale).run(q, k, v)
    expected = v[:, :64].float().mean(dim=1, keepdim=True).expand_as(q).to(q.dtype)
    assert bool(torch.isfinite(actual).all())
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)


@requires_hopper
@pytest.mark.parametrize("scale", [0.0, -0.125, 1e-7, 1e-4])
def test_signed_and_tiny_scales_against_math(scale):
    mask, _, _ = _descriptors(2, 2, 3, device="cuda")
    mask[0, 0, 1] = False
    mask[1, 1, 2] = False
    q, k, v = _inputs(2, 2, 3)
    actual = _wrapper(mask, scale=scale).run(q, k, v)
    expected = _reference(q, k, v, mask, scale)
    assert bool(torch.isfinite(actual).all())
    torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.02)


@requires_hopper
def test_offset_views_and_rejected_output_alias():
    mask, _, _ = _descriptors(1, 2, 3, device="cuda")
    wrapper = _wrapper(mask)
    q, k, v = _inputs(1, 2, 3)
    expected = wrapper.run(q, k, v)
    views = []
    for tensor in (q, k, v):
        storage = torch.empty(tensor.numel() + 1, device="cuda", dtype=tensor.dtype)
        view = storage[1:].view_as(tensor)
        view.copy_(tensor)
        views.append(view)
    storage = torch.full((q.numel() + 2,), 7, device="cuda", dtype=q.dtype)
    out = storage[1:-1].view(-1, 1, 128)
    actual = wrapper.run(*views, out=out)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert storage[0].item() == 7 and storage[-1].item() == 7
    with pytest.raises(ValueError, match="overlap"):
        wrapper.run(q, k, v, out=q.view(-1, 1, 128))
    with pytest.raises(ValueError, match="log-sum-exp"):
        wrapper.run(q, k, v, return_lse=True)
    with pytest.raises(ValueError, match="PDL"):
        wrapper.run(q, k, v, enable_pdl=True)


@requires_hopper
@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"causal": True}, "noncausal"),
        ({"pos_encoding_mode": "ALIBI"}, "positional"),
        ({"use_fp16_qk_reduction": True}, "FP32"),
        ({"logits_soft_cap": 2.0}, "logits_soft_cap"),
        ({"q_data_type": torch.float16}, "BF16"),
        ({"kv_data_type": torch.float16}, "BF16"),
        ({"sm_scale": float("nan")}, "finite"),
        ({"sm_scale": float("inf")}, "finite"),
        ({"sm_scale": 1e39}, "finite"),
    ],
)
def test_unsupported_options(kwargs, match):
    from flashinfer.sparse import VariableBlockSparseAttentionWrapper

    mask, rows, cols = _descriptors(device="cuda")
    wrapper = VariableBlockSparseAttentionWrapper(
        torch.empty(0, device="cuda", dtype=torch.uint8), backend="cake"
    )
    plan_kwargs = {"q_data_type": torch.bfloat16, **kwargs}
    with pytest.raises(ValueError, match=match):
        wrapper.plan(mask, rows, cols, 2, 2, 128, **plan_kwargs)


@requires_hopper
@pytest.mark.parametrize(
    "mutation", ["gqa", "head64", "row32", "col63", "dtype", "shape", "empty"]
)
def test_invalid_sparse_metadata(mutation):
    from flashinfer.sparse import VariableBlockSparseAttentionWrapper

    mask, rows, cols = _descriptors(device="cuda")
    h, kv_h, d = 2, 2, 128
    if mutation == "gqa":
        kv_h = 1
    elif mutation == "head64":
        d = 64
    elif mutation == "row32":
        rows[0, 0] = 32
    elif mutation == "col63":
        cols[1, 0] = 63
    elif mutation == "dtype":
        mask = mask.to(torch.int32)
    elif mutation == "shape":
        rows = rows[:, :1]
    else:
        mask[0, 1] = False
    wrapper = VariableBlockSparseAttentionWrapper(
        torch.empty(0, device="cuda", dtype=torch.uint8), backend="cake"
    )
    with pytest.raises(ValueError):
        wrapper.plan(mask, rows, cols, h, kv_h, d, q_data_type=torch.bfloat16)


@requires_hopper
def test_plan_stream_and_graph_lifetime():
    mask, rows, cols = _descriptors(1, 2, 3, device="cuda")
    producer = torch.cuda.Stream()
    with torch.cuda.stream(producer):
        wrapper = _wrapper(mask)
    # Deliberately no producer wait: plan's ready event owns that dependency.
    q, k, v = _inputs(1, 2, 3)
    expected = wrapper.run(q, k, v).clone()
    out = torch.empty((128, 1, 128), device="cuda", dtype=q.dtype)
    wrapper.run(q, k, v, out=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(q, k, v, out=out)
    graph.replay()
    torch.testing.assert_close(out.view_as(q), expected, atol=0, rtol=0)
    with pytest.raises(RuntimeError, match="captured"):
        wrapper.plan(mask, rows, cols, 1, 1, 128, q_data_type=torch.bfloat16)
    # Other wrappers must not disturb this graph's plan.
    other = _wrapper(mask)
    other.run(q, k, v)
    q.normal_()
    graph.replay()
    torch.testing.assert_close(
        out.view_as(q).float(),
        _reference(q, k, v, mask, 128**-0.5),
        atol=0.01,
        rtol=0.01,
    )
    with warnings.catch_warnings():
        # The capture is abandoned before any launch; torch warns about the empty graph.
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(ValueError, match="preallocated"):
            graph2 = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph2):
                wrapper.run(q, k, v)
