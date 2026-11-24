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

The plan is host arithmetic over Python integers: no GPU, no tensor.  It must
keep a long sequence's cuts and CTAs whatever short sequences share its batch,
run every CTA's seed rows before the rows that wait on them, never launch more
CTAs than the device can hold at once, and hand the kernel the offsets it was
planned from (the kernel refuses a batch that changed after the plan).
"""

from __future__ import annotations

import itertools

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


def _plan(lengths, H, arch, sm_count=None):
    return loader.prefill_v6_plan(
        _cu(lengths),
        num_heads=H,
        num_householder=N_HH,
        sm_count=SMS[arch] if sm_count is None else sm_count,
        arch=arch,
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
    """Every request workload is cut: the split build over every ``split_table`` row once (no pad row), as per-CTA
    lists of ``inference_csr`` on at most the SM count."""
    plan = _plan(lengths, H, arch)
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


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", sorted(SMS))
@pytest.mark.parametrize("name,H,lengths", WORKLOADS, ids=[w[0] for w in WORKLOADS])
def test_short_sequences_keep_the_long_sequences_rows_and_ctas(arch, name, H, lengths):
    """The long sequences' rows run on the same CTAs in the same order as without the one-token sequences, which
    run after them, one row per (sequence, head)."""
    shift = sum(1 for length in lengths if length == 1)
    long_lengths = lengths[shift:]
    plan = _plan(lengths, H, arch)
    alone = _plan(long_lengths, H, arch)
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
    """A long sequence of fewer rows than SMs leaves CTAs idle: the first one-token rows run there."""
    alone = _plan([8191], 10, arch)
    assert alone.grid < SMS[arch]
    plan = _plan([1, 8191], 10, arch)
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
        plan = _plan(lengths, H, arch)
        assert 1 <= plan.grid <= SMS[arch], (H, lengths)
        if not plan.split:
            continue
        per = {}
        for seq, head, _cut, _span in loader.split_table(
            _cu(lengths), N_HH, H, SMS[arch], arch=arch
        ):
            per[(seq, head)] = per.get((seq, head), 0) + 1
        for cta in _cta_lists(plan):
            kinds = [
                0 if row[2] == 0 else 1 for row in cta if per[(row[0], row[1])] > 1
            ]
            assert kinds == sorted(kinds), (H, lengths, cta)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", sorted(SMS))
def test_the_grid_never_exceeds_the_sm_count(arch):
    """The split build's seed-done wait needs every CTA resident: the grid is at most the SM count on any device
    and batch, smaller devices included."""
    for sm, H, lengths in itertools.product(
        (1, 7, 64, SMS[arch]), (1, 10, 80), [*MIXED, [65536], [1] * 15 + [8192]]
    ):
        plan = _plan(lengths, H, arch, sm_count=sm)
        assert 1 <= plan.grid <= sm
        if plan.split:
            assert len(plan.offsets) == plan.grid + 1
            assert plan.offsets[-1] == len(plan.rows)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", sorted(SMS))
def test_short_batches_take_the_block_build(arch):
    # 64 sequences of 256 tokens (12 chunks each): no sequence is cut, so the block build runs, longest first
    plan = _plan([256] * 64, 10, arch)
    assert not plan.split and plan.grid == SMS[arch]
    assert plan.rows == tuple(
        (seq, head, 0, 4) for seq in range(64) for head in range(10)
    )
    plan = _plan([1, 300, 64, 1], 2, arch)
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
    plan = _plan([0, 300], 4, arch)
    assert plan.split
    assert (0, 0, 0, 0) in plan.rows  # the state copy of the empty sequence


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
