# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The launch plan of the Cake GDP Blackwell prefill body (``cake_jit``, the BLACKWELL V6 LAUNCH PLAN block).

The plan is host arithmetic over Python integers: no GPU, no tensor.  The plan
a call launches is the PLAN CHOOSER's pick among candidate plans of the batch:
a deterministic, pure (cached) function of the host offsets, the head count, n,
the SM count and the arch, the plan of record unless another candidate is
modeled faster by the margin.  Every candidate must run every CTA's seed rows
before the rows that wait on them, never launch more CTAs than the device can
hold at once, run every row of its table once, and hand the kernel the offsets
it was planned from (the kernel refuses a batch that changed after the plan).
The plan of record keeps a long sequence's cuts and CTAs whatever short
sequences share its batch; a chosen candidate may cut it otherwise (the
inference builds agree with the exact recurrence within FlashInfer's tolerance
for any cut: ``tests/gdp/test_cake_gdp_prefill_bw.py``).
"""

from __future__ import annotations

import itertools
import random

import pytest

from tests.cake.test_helpers.cake_package import cake_package

loader = cake_package("cake_gdp", "cake_jit")

SMS = {"sm_100a": 148, "sm_103a": 152}  # B200 / GB300
N_HH = 3
# The prefill request's workloads: d one-token sequences followed by one long sequence (A), one long sequence (B),
# two long sequences (C).
A_HEADS = (10, 20, 40, 80)
A_LENGTHS = (8192, 16384, 32768, 65536)
A_SHORT = (0, 1, 4, 8, 15)
# Further batches: 1-token, short, empty and long sequences in any order, many medium sequences, all short.
MIXED = (
    [1000, 1, 70000, 3, 0, 4096],
    [65, 129, 8192, 8192],
    [5000] * 7,
    [1] * 40 + [20000] + [1] * 3,
    [0, 1, 0, 9000],
    [256] * 64,
)


def _cu(lengths):
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    return cu


def _workloads():
    for H in A_HEADS:
        for L in A_LENGTHS:
            for d in A_SHORT:
                yield f"A_H{H}_L{L}_d{d}", H, [1] * d + [min(L, 65536 - d)]
        yield f"B_H{H}", H, [65536]
        yield f"C_H{H}", H, [36864, 28672]


WORKLOADS = list(_workloads())


def _kw(H, arch, sm_count=None):
    return dict(
        num_heads=H,
        num_householder=N_HH,
        sm_count=SMS[arch] if sm_count is None else sm_count,
        arch=arch,
    )


def _plan(lengths, H, arch, sm_count=None):
    return loader.prefill_v6_plan(_cu(lengths), **_kw(H, arch, sm_count))


def _record(lengths, H, arch, sm_count=None):
    return loader.prefill_v6_record_plan(_cu(lengths), **_kw(H, arch, sm_count))


def _union_cells():
    """The 147 shapes of the acceptance union set: the request's 88, a grid of 1K-32K prefills with up to 16
    one-token sequences (H 24 / 40 / 80) and 17 shapes of the short-prefill report."""
    cells = {name: (H, lengths) for name, H, lengths in WORKLOADS}

    def a_cell(H, L, d):
        cells.setdefault(f"A_H{H}_L{L}_d{d}", (H, [1] * d + [min(L, 65536 - d)]))

    for H in (24, 40, 80):
        for L in (8192, 16384, 32768):
            a_cell(H, L, 0)
        for L in (1024, 2048, 4096):
            for d in (1, 2, 4, 8, 16):
                a_cell(H, L, d)
    for H, L, d in (
        (20, 1024, 1),
        (20, 1024, 4),
        (20, 1024, 16),
        (20, 1024, 32),
        (20, 1024, 64),
        (20, 2048, 16),
        (20, 2048, 32),
        (24, 1024, 1),
        (24, 1024, 16),
        (24, 1024, 32),
        (24, 1024, 64),
        (24, 2048, 16),
        (24, 2048, 32),
        (40, 1024, 8),
        (40, 1024, 16),
        (80, 4096, 0),
        (80, 65536, 0),
    ):
        a_cell(H, L, d)
    return cells


UNION = _union_cells()


def _random_batches(seed, count):
    """Random batches: one-token, empty, short and long sequences in any order, 1 to 80 heads."""
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        lengths = []
        for _seq in range(rng.randint(1, 24)):
            kind = rng.random()
            if kind < 0.35:
                lengths.append(1)
            elif kind < 0.42:
                lengths.append(0)
            elif kind < 0.75:
                lengths.append(rng.randint(2, 3000))
            else:
                lengths.append(rng.randint(3000, 40000))
        if sum(lengths) == 0:
            lengths.append(rng.randint(1, 5000))
        out.append((rng.choice((1, 3, 8, 10, 13, 20, 24, 40, 64, 80)), lengths))
    return out


RANDOM = _random_batches(7, 120)


def _old_inference_csr(rows, cu_seqlens, n, grid):
    """``inference_csr`` as FlashInfer shipped it before the plan chooser (one row at a time, ``O(R G)``): the
    reference of the bulk placement."""
    raw = [int(x) for x in cu_seqlens]
    total = raw[-1]
    chunks = [
        loader.split_chunks(e - s, n)
        for s, e in (
            loader.split_sequence_bounds(raw[q], raw[q + 1], total)
            for q in range(len(raw) - 1)
        )
    ]
    per = {}
    for seq, head, _cut, _span in rows:
        per[(seq, head)] = per.get((seq, head), 0) + 1
    split_seqs = {seq for (seq, _head), count in per.items() if count > 1}
    lists = [[] for _ in range(grid)]
    load = [0] * grid
    for at, row in enumerate(r for r in rows if r[0] in split_seqs):
        lists[at % grid].append(row)
        load[at % grid] += loader.inference_row_cost(row, chunks[row[0]], n)
    for b in range(grid):
        lists[b].sort(key=lambda r: 0 if r[2] == 0 else 1)
    top = max(load, default=0)
    short = sorted(
        (r for r in rows if r[0] not in split_seqs),
        key=lambda r: -loader.inference_row_cost(r, chunks[r[0]], n),
    )
    for row in short:
        cost = loader.inference_row_cost(row, chunks[row[0]], n)
        room = [b for b in range(grid) if load[b] + cost <= top]
        b = min(room or range(grid), key=lambda c: (load[c], c))
        lists[b].append(row)
        load[b] += cost
        top = max(top, load[b])
    offsets = [0, *itertools.accumulate(len(x) for x in lists)]
    return [r for x in lists for r in x], offsets


def _any_plan(name, lengths, H, arch, sm_count=None):
    """The launched plan (``"chosen"``) or candidate ``name``'s plan, None when the candidate does not apply."""
    if name == "chosen":
        return _plan(lengths, H, arch, sm_count)
    if name not in loader.prefill_v6_plan_candidates(
        _cu(lengths), **_kw(H, arch, sm_count)
    ):
        return None
    return loader.prefill_v6_candidate_plan(
        name, _cu(lengths), **_kw(H, arch, sm_count)
    )


def _cta_lists(plan):
    """Each CTA's rows in the order it runs them: the split build's per-CTA lists, the block build's stride."""
    if plan.split:
        o = plan.offsets
        return [list(plan.rows[o[b] : o[b + 1]]) for b in range(plan.grid)]
    return [list(plan.rows[b :: plan.grid]) for b in range(plan.grid)]


@pytest.mark.cpu_only
def test_the_request_has_88_workloads():
    assert len(WORKLOADS) == 88


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", sorted(SMS))
@pytest.mark.parametrize("name,H,lengths", WORKLOADS, ids=[w[0] for w in WORKLOADS])
def test_the_cta_lists_hold_the_split_rules_of_record(arch, name, H, lengths):
    """Every request workload is cut by the plan of record: the split build over every ``split_table`` row once (no
    pad row), as per-CTA lists of ``inference_csr`` on at most the SM count, the lists the one-row-at-a-time rule
    gives."""
    plan = _record(lengths, H, arch)
    table = loader.split_table(_cu(lengths), N_HH, H, SMS[arch], arch=arch)
    assert plan.split and loader.launch_split(table, arch)
    assert plan.grid == min(SMS[arch], len(table))
    assert sorted(plan.rows) == sorted(table)
    offsets = plan.offsets
    assert len(offsets) == plan.grid + 1 and offsets[0] == 0
    assert offsets[-1] == len(plan.rows)
    assert all(a <= b for a, b in itertools.pairwise(offsets))
    rows, csr = loader.inference_csr(table, _cu(lengths), N_HH, plan.grid)
    assert (list(plan.rows), list(offsets)) == (rows, csr)
    assert (rows, csr) == _old_inference_csr(table, _cu(lengths), N_HH, plan.grid)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", sorted(SMS))
@pytest.mark.parametrize("name,H,lengths", WORKLOADS, ids=[w[0] for w in WORKLOADS])
def test_short_sequences_keep_the_long_sequences_rows_and_ctas_in_the_plan_of_record(
    arch, name, H, lengths
):
    """In the plan of record the long sequences' rows run on the same CTAs in the same order as without the one-token
    sequences, which run after them, one row per (sequence, head).  (The plan a call launches is chosen per batch and
    may cut the long sequence otherwise: see ``test_the_plan_is_a_pure_function_of_the_batch``.)"""
    shift = sum(1 for length in lengths if length == 1)
    long_lengths = lengths[shift:]
    plan = _record(lengths, H, arch)
    alone = _record(long_lengths, H, arch)
    assert plan.split == alone.split
    mine = _cta_lists(plan)
    theirs = _cta_lists(alone)
    assert plan.grid >= alone.grid
    for b in range(plan.grid):
        lead = [
            (seq - shift, head, cut, span)
            for seq, head, cut, span in mine[b]
            if seq >= shift
        ]
        assert lead == (theirs[b] if b < alone.grid else []), b
        tail = [row for row in mine[b] if row[0] < shift]
        assert mine[b][len(mine[b]) - len(tail) :] == tail, b  # the short rows run last
    short = sorted((row[0], row[1]) for cta in mine for row in cta if row[0] < shift)
    assert short == [(seq, head) for seq in range(shift) for head in range(H)]


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", sorted(SMS))
def test_short_rows_take_the_idle_ctas_first(arch):
    """A long sequence of fewer rows than SMs leaves CTAs idle: in the plan of record the first one-token rows run
    there."""
    alone = _record([8191], 10, arch)
    assert alone.grid < SMS[arch]
    plan = _record([1, 8191], 10, arch)
    lists = _cta_lists(plan)
    idle = range(alone.grid, plan.grid)
    assert idle and all([row[0] for row in lists[b]] == [0] for b in idle)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", sorted(SMS))
def test_every_cta_runs_its_seed_rows_first(arch):
    """On each CTA the cut-0 rows of the cut sequences come first: an owner waits only on a cut-0 row (which waits
    on nothing), and every CTA is resident at once, so no launch can deadlock."""
    batches = [lengths for _name, _H, lengths in WORKLOADS] + list(MIXED)
    for H, lengths in itertools.product((1, 3, 10, 24, 80), batches):
        for name in ("chosen", *loader.V6_PLAN_CANDIDATES):
            plan = _any_plan(name, lengths, H, arch)
            if plan is None:
                continue
            assert 1 <= plan.grid <= SMS[arch], (H, lengths, name)
            if not plan.split:
                continue
            per = {}
            for seq, head, _cut, _span in plan.rows:
                per[(seq, head)] = per.get((seq, head), 0) + 1
            for cta in _cta_lists(plan):
                kinds = [
                    0 if row[2] == 0 else 1
                    for row in cta
                    if per[(row[0], row[1])] > 1
                ]
                assert kinds == sorted(kinds), (H, lengths, name, cta)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", sorted(SMS))
def test_the_grid_never_exceeds_the_sm_count(arch):
    """The split build's seed-done wait needs every CTA resident: the grid is at most the SM count on any device
    and batch, smaller devices included."""
    for sm, H, lengths in itertools.product(
        (1, 7, 64, SMS[arch]), (1, 10, 80), [*MIXED, [65536], [1] * 15 + [8192]]
    ):
        for name in ("chosen", *loader.V6_PLAN_CANDIDATES):
            plan = _any_plan(name, lengths, H, arch, sm)
            if plan is None:
                continue
            assert 1 <= plan.grid <= sm, (sm, H, lengths, name)
            if plan.split:
                assert len(plan.offsets) == plan.grid + 1
                assert plan.offsets[-1] == len(plan.rows)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", sorted(SMS))
def test_short_batches_take_the_block_build_in_the_plan_of_record(arch):
    # 64 sequences of 256 tokens (12 chunks each): no sequence is cut, so the block build runs, longest first
    plan = _record([256] * 64, 10, arch)
    assert not plan.split and plan.grid == SMS[arch]
    assert plan.rows == tuple(
        (seq, head, 0, 4) for seq in range(64) for head in range(10)
    )
    plan = _record([1, 300, 64, 1], 2, arch)
    assert not plan.split and plan.grid == 8
    assert plan.rows == (
        (1, 0, 0, 5),
        (1, 1, 0, 5),
        (0, 0, 0, 1),
        (0, 1, 0, 1),
        (2, 0, 0, 1),
        (2, 1, 0, 1),
        (3, 0, 0, 1),
        (3, 1, 0, 1),
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", sorted(SMS))
def test_an_empty_sequence_takes_the_split_build(arch):
    for plan in (_plan([0, 300], 4, arch), _record([0, 300], 4, arch)):
        assert plan.split
        assert (0, 0, 0, 0) in plan.rows  # the state copy of the empty sequence
    choices = loader.prefill_v6_plan_candidates(_cu([0, 300]), **_kw(4, arch))
    assert "block" not in choices and "blockbal" not in choices


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", sorted(SMS))
def test_every_plan_names_a_shipped_program(arch):
    """The registry's Blackwell route ships, per state form (the dense FP32 programs and the FP32, BF16 and FP16
    pool programs), one program with and one without the split, all for n = 3; the route refuses every other n."""
    forms = loader.v6_state_forms(arch)
    assert forms == {("f32", False), ("f32", True), ("bf16", True), ("f16", True)}
    names = set(loader.ROUTES[arch][loader.PREFILL_ROUTE])
    shipped = set()
    for state_dtype, pool in forms:
        builds = loader.v6_route_builds(arch, state_dtype, pool)
        assert sorted(builds) == [False, True], f"{arch} ships {builds}"
        for split, name in builds.items():
            assert loader.v6_state_form(name) == (state_dtype, pool)
            assert loader.MODULES[name]["select_rule"]["split_items"] == split
        shipped |= set(builds.values())
    assert shipped == names
    for name in names:
        assert loader.MODULES[name]["body"] == loader.V6_BODY
        assert loader.MODULES[name]["arches"] == [arch]
    assert loader.v6_householders(arch) == {N_HH}
    assert loader.v6_route_builds(arch) == loader.v6_route_builds(arch, "f32", False)
    capability = {"sm_100a": (10, 0), "sm_103a": (10, 3)}[arch]
    for n in (1, 2, 3, 4):
        assert loader.prefill_route_applies(
            device_capability=capability, num_householder=n
        ) == (n == N_HH)
    for lengths, H in (([65536], 80), ([1] * 4 + [8192], 10), ([256] * 64, 10)):
        plan = _plan(lengths, H, arch)
        name = loader.selected_v6_module(arch, plan)
        assert loader.MODULES[name]["select_rule"]["split_items"] == plan.split
        for state_dtype, pool in forms:
            name = loader.selected_v6_module(
                arch, plan, state_dtype=state_dtype, pool=pool
            )
            assert loader.v6_state_form(name) == (state_dtype, pool)
            assert loader.MODULES[name]["select_rule"]["split_items"] == plan.split
    with pytest.raises(ValueError, match="state_dtype='bf16' and pool=False"):
        loader.selected_v6_module(
            arch, _plan([64], 4, arch), state_dtype="bf16", pool=False
        )


_POOL_KEYS = ("use_state_indices", "use_output_state_indices")
_POOL_LOGICAL = (
    "state_indices",
    "output_state_indices",
    "initial_state_stride_slot",
    "output_state_stride_slot",
)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", sorted(SMS))
def test_the_pool_programs_are_twins_of_the_dense_programs(arch):
    """Every pool program is the dense FP32 program of its split with both slot tables, and every BF16 / FP16
    program the FP32 pool program of its split with its ``state_dtype``: the same build otherwise, the same launch.
    The pool programs bind the slot tables and the slot strides; the 16-bit ones the seed and the round count."""
    dense = loader.v6_route_builds(arch, "f32", False)
    f32_pool = loader.v6_route_builds(arch, "f32", True)
    for split in (False, True):
        base = loader.MODULES[dense[split]]
        pool = loader.MODULES[f32_pool[split]]
        assert pool["build"] == {**base["build"], **{key: True for key in _POOL_KEYS}}
        assert (
            pool["launch"] == base["launch"] and pool["grid_rule"] == base["grid_rule"]
        )
        assert all(pool["logical"][name] == name for name in _POOL_LOGICAL)
        assert not any(name in base["logical"] for name in _POOL_LOGICAL)
        for state_dtype in ("bf16", "f16"):
            low = loader.MODULES[loader.v6_route_builds(arch, state_dtype, True)[split]]
            assert low["build"] == {**pool["build"], "state_dtype": state_dtype}
            assert low["launch"] == pool["launch"]
            assert low["logical"] == {
                **pool["logical"],
                "rand_seed": "rand_seed",
                "philox_rounds": "philox_rounds",
            }
            assert low["arg_plan"][: len(pool["arg_plan"]) - 3] == pool["arg_plan"][:-3]


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", sorted(SMS))
def test_the_items_carry_the_offsets_the_plan_was_made_from(arch):
    """The ``items`` words: the rows, then (split build) the CTAs' row offsets, then the planned sequence count and
    offsets, which the kernel compares with the device ``cu_seqlens`` before it runs any row."""
    for lengths, H in (([1] * 4 + [8192], 10), ([256] * 64, 10), ([0, 300], 4)):
        cu = _cu(lengths)
        plan = _plan(lengths, H, arch)
        build = loader.MODULES[loader.selected_v6_module(arch, plan)]["build"]
        assert build["plan_guard"] and bool(build["csr"]) == plan.split
        words = loader.v6_item_words(arch, plan, cu)
        expect = [w for row in plan.rows for w in row]
        if plan.split:
            expect += plan.offsets
        assert list(words) == [*expect, len(lengths), *cu]
    with pytest.raises(ValueError, match="row offsets"):
        loader.v6_item_words(
            arch, _plan([0, 300], 4, arch)._replace(offsets=None), [0, 0, 300]
        )


@pytest.mark.cpu_only
def test_the_sm90a_split_tables_are_unchanged_by_the_arch_keyed_rules():
    """The arch-keyed rules default to sm_90a: its tables keep the Hopper floors and no balanced cut."""
    for H, L in itertools.product((10, 24, 80), (1024, 8192, 32768, 65536)):
        table = loader.split_table([0, L], N_HH, H, 132)
        assert table == loader.split_table([0, L], N_HH, H, 132, arch="sm_90a")
        assert all(span > 0 for _seq, _head, _cut, span in table)
    assert (
        loader.split_span_min(20) == 24 and loader.split_span_min(20, "sm_100a") == 48
    )


# ---------------------------------------------------------------------------
# the plan chooser
# ---------------------------------------------------------------------------


def _naive_lpt(loads, cost, k):
    """One row at a time to the CTA of least (load, index): the CTA of each row."""
    loads = list(loads)
    out = []
    for _ in range(k):
        b = min(range(len(loads)), key=lambda c: (loads[c], c))
        out.append(b)
        loads[b] += cost
    return out


@pytest.mark.cpu_only
def test_the_bulk_placement_is_the_one_row_at_a_time_rule():
    """``lpt_rounds`` / ``lpt_counts`` / ``lpt_place`` give each row the CTA the one-row-at-a-time least-load rule
    gives it (3000 random cases: idle and loaded CTAs, few and many rows, zero cost)."""
    rng = random.Random(0)
    for case in range(3000):
        G = rng.randint(1, 160)
        spread = rng.choice((0, 50, 900, 20000))
        loads = [rng.choice((0, rng.randint(0, spread))) for _ in range(G)]
        cost = rng.choice((0, 1, rng.randint(1, 900)))
        k = rng.randint(0, rng.choice((3, G, 6 * G)))
        want = _naive_lpt(loads, cost, k)
        head, phase, full, tail = loader.lpt_rounds(loads, cost, k)
        assert head + phase * full + tail == want, case
        counts = [0] * G
        for b in want:
            counts[b] += 1
        assert loader.lpt_counts(loads, cost, k) == counts, case
        lists = [[("x", b)] for b in range(G)]
        placed = list(loads)
        loader.lpt_place(placed, cost, list(range(k)), lists)
        assert placed == [load + x * cost for load, x in zip(loads, counts)], case
        for b in range(G):
            assert lists[b] == [("x", b), *(i for i in range(k) if want[i] == b)], case


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", sorted(SMS))
def test_inference_csr_is_the_one_row_at_a_time_rule(arch):
    """The bulk ``inference_csr`` returns the lists of the per-row rule it replaced, on random batches and devices."""
    rng = random.Random(1)
    for H, lengths in [*((H, lengths) for H, lengths in UNION.values()), *RANDOM]:
        sm = rng.choice((7, 64, SMS[arch]))
        table = loader.split_table(_cu(lengths), N_HH, H, sm, arch=arch)
        if not table:
            continue
        grid = max(1, min(sm, len(table)))
        assert loader.inference_csr(table, _cu(lengths), N_HH, grid) == _old_inference_csr(
            table, _cu(lengths), N_HH, grid
        ), (H, lengths, sm)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", sorted(SMS))
def test_the_explicit_floor_rules_at_the_record_floor_are_the_rules_of_record(arch):
    for H, sm in itertools.product((1, 10, 12, 13, 20, 40, 80), (7, 148, 152)):
        floor = loader.split_span_min(H, arch)
        for chunks in (*range(0, 400), *range(400, 6200, 37)):
            assert loader.v6_split_count(
                chunks, H, N_HH, sm, floor
            ) == loader.split_count(chunks, H, N_HH, sm, arch=arch), (H, sm, chunks)
            assert loader.v6_split_bal_slot(
                chunks, H, N_HH, arch, floor
            ) == loader.split_bal_slot(chunks, H, N_HH, arch=arch), (H, chunks)


def _all_batches():
    yield from ((H, lengths) for _name, H, lengths in WORKLOADS)
    yield from UNION.values()
    yield from itertools.product((1, 3, 10, 24, 80), MIXED)
    yield from RANDOM


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", sorted(SMS))
def test_the_main_candidate_is_the_plan_of_record(arch):
    for H, lengths in _all_batches():
        main = loader.prefill_v6_candidate_plan("main", _cu(lengths), **_kw(H, arch))
        assert main == _record(lengths, H, arch), (H, lengths)


def _candidate_table(name, lengths, H, arch):
    """The rows candidate ``name`` must run, from the rules: the split rules with the least span of the candidate's
    floor (V8-BAL or not), no cut (``exact``), or one block row per non-empty sequence and head."""
    if name in ("block", "blockbal"):
        return sorted(
            (q, h, 0, -(-length // 64))
            for q, length in enumerate(lengths)
            if length
            for h in range(H)
        )
    if name == "main":
        return sorted(loader.split_table(_cu(lengths), N_HH, H, SMS[arch], arch=arch))
    if name == "exact":
        return sorted(
            loader.split_table(_cu(lengths), N_HH, H, SMS[arch], "exact", arch=arch)
        )
    floor = int(name[1:].removesuffix("nb"))
    span_min = min(floor, loader.split_span_min(H, arch))
    rows = []
    for q, length in enumerate(lengths):
        chunks = loader.split_chunks(length, N_HH)
        slot = 0
        if not name.endswith("nb") and chunks >= loader.SPLIT_MIN_CHUNKS:
            slot = loader.v6_split_bal_slot(chunks, H, N_HH, arch, span_min)
        if slot:
            rows.extend(loader.split_bal_rows(q, chunks, H, N_HH, slot, arch))
            continue
        p, span = loader.v6_split_count(chunks, H, N_HH, SMS[arch], span_min)
        rows.extend((q, h, j * span, span) for j in range(p) for h in range(H))
    return sorted(rows)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", sorted(SMS))
def test_every_candidate_runs_its_table_once_on_its_build(arch):
    """Each candidate's plan runs every row of its table once: per-CTA lists (split build) or the stride layout of
    the block build's rows (a block candidate only without an empty sequence)."""
    for H, lengths in _all_batches():
        choices = loader.prefill_v6_plan_candidates(_cu(lengths), **_kw(H, arch))
        assert ("block" in choices) == (0 not in lengths)
        for name, (alias, _cost) in choices.items():
            plan = loader.prefill_v6_candidate_plan(name, _cu(lengths), **_kw(H, arch))
            assert sorted(plan.rows) == _candidate_table(alias, lengths, H, arch), (
                name,
                H,
                lengths,
            )
            if name in ("block", "blockbal") or (name == "main" and alias == "block"):
                assert not plan.split and plan.offsets is None
            else:
                assert plan.split and len(plan.offsets) == plan.grid + 1
            assert plan.grid == max(1, min(SMS[arch], len(plan.rows)))


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", sorted(SMS))
def test_the_modeled_time_is_the_most_loaded_cta_of_the_plan(arch):
    """A candidate's modeled time (computed from grouped rows) is its fixed cost plus the most loaded CTA of its plan
    under the arch's costs, and never below its lower bound."""
    costs = loader.V6_PLAN_COSTS[arch]
    for H, lengths in _all_batches():
        cu = _cu(lengths)
        batch = loader._V6Batch(tuple(cu), H, N_HH, SMS[arch], arch, cu[-1])
        cands, _alias = loader._v6_candidates(batch)
        for cand in cands:
            cost = cand.cost()
            assert cost >= cand.bound(), (cand.name, H, lengths)
            plan = cand.plan()
            loads = []
            for cta in _cta_lists(plan):
                if plan.split:
                    loads.append(
                        sum(
                            loader._v6_split_row_cost(
                                row, batch.chunks[row[0]], batch.lens[row[0]], N_HH, costs
                            )
                            for row in cta
                        )
                    )
                else:
                    loads.append(
                        sum(
                            loader._v6_block_row_cost(batch.lens[row[0]], N_HH, costs)
                            for row in cta
                        )
                    )
            fixed = costs["fixed"] if plan.split else costs["block_fixed"]
            assert cost == fixed + max(loads), (cand.name, H, lengths)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", sorted(SMS))
def test_the_chooser_takes_the_least_modeled_candidate(arch):
    """The launched plan is the candidate of least modeled time (the first in V6_PLAN_CANDIDATES order on a tie) when
    it is V6_PLAN_MARGIN_PERCENT below the plan of record's, else the plan of record; every candidate modeled, no
    pruning."""
    order = {name: at for at, name in enumerate(loader.V6_PLAN_CANDIDATES)}
    margin = loader.V6_PLAN_MARGIN_PERCENT
    picks = set()
    for H, lengths in _all_batches():
        cu = _cu(lengths)
        choices = loader.prefill_v6_plan_candidates(cu, **_kw(H, arch))
        ref = choices["main"][1]
        distinct = {alias: cost for alias, cost in choices.values()}
        # the distinct candidates in the chooser's order: the split specs, main, block, blockbal
        best = min(distinct, key=lambda name: (distinct[name], order[name]))
        want = best if 100 * distinct[best] < (100 - margin) * ref else choices["main"][0]
        got = loader.prefill_v6_plan_choice(cu, **_kw(H, arch))
        assert got == want, (H, lengths, choices)
        assert _plan(lengths, H, arch) == loader.prefill_v6_candidate_plan(
            got, cu, **_kw(H, arch)
        )
        picks.add(got)
    assert {"main", "f12", "f36"} <= picks, picks


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", sorted(SMS))
@pytest.mark.parametrize(
    "name,pick",
    [
        ("A_H20_L1024_d16", "f12"),
        ("A_H24_L1024_d1", "f12"),
        ("A_H40_L1024_d8", "f18"),
        ("A_H80_L4096_d0", "f36"),
        ("A_H80_L65536_d0", "main"),
        ("B_H80", "main"),
    ],
)
def test_the_calibrated_picks_of_the_short_prefill_report(arch, name, pick):
    """The picks of record on shapes of the short-prefill report: a 1K-4K prefill is cut below floor 48 (the plan of
    record runs it as one serial chain per head), a full-grid 64K prefill keeps the plan of record."""
    H, lengths = UNION[name]
    assert loader.prefill_v6_plan_choice(_cu(lengths), **_kw(H, arch)) == pick


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", sorted(SMS))
def test_the_plan_is_a_pure_function_of_the_batch(arch):
    """The launched plan depends on the host offsets, the head count, n, the SM count, the arch and the token count
    only: the same plan in any call order, from a list or a tuple, with or without the plan cache, and the same as
    the uncached pick."""
    batches = [*UNION.values(), *RANDOM[:40]]
    loader._v6_plan_cached.cache_clear()
    first = [_plan(lengths, H, arch) for H, lengths in batches]
    loader._v6_plan_cached.cache_clear()
    shuffled = list(range(len(batches)))
    random.Random(3).shuffle(shuffled)
    again = {i: _plan(batches[i][1], batches[i][0], arch) for i in shuffled}
    for i, (H, lengths) in enumerate(batches):
        assert again[i] == first[i]
        cu = _cu(lengths)
        assert loader.prefill_v6_plan(tuple(cu), **_kw(H, arch)) == first[i]
        assert loader.prefill_v6_plan(
            cu, total_tokens=cu[-1], **_kw(H, arch)
        ) == first[i]
        name = loader.prefill_v6_plan_choice(cu, **_kw(H, arch))
        assert loader.prefill_v6_candidate_plan(name, cu, **_kw(H, arch)) == first[i]


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", sorted(SMS))
def test_the_plan_cache_keeps_the_least_recently_used_plans(arch):
    """A repeated batch is a cache hit returning the same plan object; the key holds every argument; the cache keeps
    V6_PLAN_CACHE_SIZE plans."""
    cached = loader._v6_plan_cached
    cached.cache_clear()
    lengths = [1] * 16 + [1024]
    plan = _plan(lengths, 20, arch)
    hits = cached.cache_info().hits
    assert _plan(lengths, 20, arch) is plan
    assert loader.prefill_v6_plan(tuple(_cu(lengths)), **_kw(20, arch)) is plan
    assert cached.cache_info().hits == hits + 2
    others = (
        loader.prefill_v6_plan(_cu(lengths), **_kw(24, arch)),
        loader.prefill_v6_plan(_cu(lengths), **_kw(20, arch, sm_count=64)),
        loader.prefill_v6_plan(
            _cu(lengths), total_tokens=sum(lengths) - 1, **_kw(20, arch)
        ),
    )
    assert all(other is not plan for other in others)
    assert cached.cache_info().maxsize == loader.V6_PLAN_CACHE_SIZE
    for extra in range(loader.V6_PLAN_CACHE_SIZE + 4):
        _plan([1] * (extra % 7) + [2000 + extra], 20, arch)
    assert cached.cache_info().currsize == loader.V6_PLAN_CACHE_SIZE
    assert _plan(lengths, 20, arch) == plan  # recomputed after eviction, equal
