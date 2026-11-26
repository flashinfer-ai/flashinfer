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

CPU-only tests of the Cake Gated DeltaProduct (GDP) prefill body route of
record (``cake_jit``, the BODY ROUTE OF RECORD block): the per-sequence choice
between the V6 fork-order body and the merged chunked body, the launch plan of
a batch, the plan of the V6 warm-up split programs (the SM90 V6 WARM-UP SPLIT
PLAN block), the sm_90a ``prefill`` registry and the identities of its records.
The route is host arithmetic over Python integers: no GPU, no tensor, no
device fixture.  ``test_cake_gdp_prefill.py`` runs the kernels.
"""

from __future__ import annotations

import inspect
import itertools
import re

import pytest

from tests.cake.test_helpers.cake_package import cake_package

cake_build = cake_package("cake_gdp", "cake_build")
loader = cake_package("cake_gdp", "cake_jit")

P3 = loader.ROUTE_BODY_P3
V6 = loader.ROUTE_BODY_V6
ARCH = "sm_90a"
SM_COUNT = 132  # H100 SXM / H200
N_HH = 3  # num_householder of the performance rows
# The 18 prefill performance rows: (H) x (N, L) at n = 3; every row is a uniform batch.  The merged class takes the
# single long sequences at H = 10 and H = 24 (its split items are the cheaper); every other row is V6-class.
PERF_HEADS = (80, 10, 24)
PERF_NL = ((1, 1024), (1, 8192), (1, 32768), (4, 4096), (16, 1024), (64, 256))
MERGED_ROWS = {(10, 1, 8192), (10, 1, 32768), (24, 1, 8192), (24, 1, 32768)}
HEADS = (1, 2, 3, 6, 10, 12, 20, 24, 40, 80)
# The logical names only the Blackwell body's records take (its item table, workspace, flags word and counts; the
# write slot table and the rounding seed and rounds of its pool forms).
BLACKWELL_ONLY_LOGICAL = {
    "items",
    "split_workspace",
    "state_stride_slot",
    "num_items",
    "flags",
    "output_state_indices",
    "rand_seed",
    "philox_rounds",
}
SM_COUNTS = (132, 114, 148, 152)
# The sm_90a prefill route: the three V6 programs (plain, initial state, initial state + slot pool), the two unit
# programs and the three split programs of the merged body, in route order.
V6_NAMES = (
    "cake_gdp_0de8fd0ee57885709a1c",
    "cake_gdp_c81addff1e986f49ec99",
    "cake_gdp_55a7b3c2baf155618309",
)
UNIT_NAMES = ("cake_gdp_c8580552e73f2264f8eb", "cake_gdp_f39daf3dd3cf89844fc1")
SPLIT_NAMES = (
    "cake_gdp_3220d37914678385a837",
    "cake_gdp_245414df8449fb1825a1",
    "cake_gdp_b81854850bf4b769295c",
)
# The warm-up split twins of the three V6 programs (the V6 body's ``split_items`` programs).
V6_SPLIT_NAMES = (
    "cake_gdp_5acbda2bd1002fd68a95",
    "cake_gdp_edfb71d95c7a9c2a133d",
    "cake_gdp_446ecf26f72f95aacf84",
)
# (use_initial_state, use_state_indices, store_final_state, unit_gates, unit_beta) per program group.
V6_FLAGS = (
    (False, False, True, False, False),
    (True, False, True, False, False),
    (True, True, True, False, False),
)
UNIT_FLAGS = (
    (False, False, True, True, False),
    (False, False, True, False, True),
)
SPLIT_FLAGS = V6_FLAGS
# The merged body's exact gate-tensor programs the route no longer carries: plain, h0, pool, no-final.
REMOVED_FLAGS = (
    (False, False, True, False, False),
    (True, False, True, False, False),
    (True, True, True, False, False),
    (False, False, False, False, False),
)
MIXED_LENS = (8192, 1000, 0, 8192)


def _offsets(lens: tuple[int, ...]) -> tuple[int, ...]:
    return (0, *itertools.accumulate(lens))


def _classes(
    lens: tuple[int, ...], H: int, n: int = N_HH, sm_count: int = SM_COUNT, **flags
) -> list[str]:
    return loader.prefill_body_classes(_offsets(lens), n, H, sm_count, **flags)


def _select(body: str, flags: tuple[bool, ...], **extra: bool) -> str:
    use_initial, use_indices, store, unit_g, unit_beta = flags
    return loader.selected_prefill_module(
        ARCH,
        body=body,
        use_initial_state=use_initial,
        use_state_indices=use_indices,
        store_final_state=store,
        unit_gates=unit_g,
        unit_beta=unit_beta,
        **extra,
    )


# ---------------------------------------------------------------------------
# The performance rows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("H", PERF_HEADS)
@pytest.mark.parametrize("N,L", PERF_NL)
def test_performance_rows_are_uniform_batches(H: int, N: int, L: int) -> None:
    """Every performance row is one launch: the merged class exactly on the single long sequences at H = 10 and
    H = 24, the V6 class everywhere else."""
    classes = _classes((L,) * N, H)
    body = P3 if (H, N, L) in MERGED_ROWS else V6
    assert classes == [body] * N, (H, N, L, classes)
    assert loader.prefill_launch_plan(classes) == [(body, list(range(N)))]


def test_pinned_body_costs() -> None:
    """The costs of record of a 384-chunk sequence (L = 8192 at n = 3) on 132 SMs, in tenths of a V6 chunk-step:
    the merged split items win at H = 10 and lose at H = 80 (38 and 272 item waves at 3.6 V6 steps each); an empty sequence costs nothing under either body."""
    assert loader.prefill_body_costs(384, 10, N_HH, SM_COUNT) == (3840, 1368)
    assert loader.prefill_body_costs(384, 80, N_HH, SM_COUNT) == (3840, 9792)
    assert loader.prefill_body_costs(0, 10, N_HH, SM_COUNT) == (0, 0)
    assert loader.ROUTE_MIN_CHUNKS_P3 == 384
    assert loader.ROUTE_BODIES == (P3, V6)


# ---------------------------------------------------------------------------
# Class properties
# ---------------------------------------------------------------------------

CHUNK_COUNTS = (0, 2, 12, 48, 192, 382, 384, 386, 400, 512, 768, 1024, 1536, 4096)


@pytest.mark.parametrize("unit_gates,unit_beta", ((True, False), (False, True)))
def test_unit_calls_are_merged_class_at_every_length(
    unit_gates: bool, unit_beta: bool
) -> None:
    for chunks, H, n, sm_count in itertools.product(
        CHUNK_COUNTS, HEADS, loader.NUM_HOUSEHOLDER, SM_COUNTS
    ):
        assert (
            loader.prefill_body_class(
                chunks, H, n, sm_count, unit_gates=unit_gates, unit_beta=unit_beta
            )
            == P3
        ), (chunks, H, n, sm_count)


def test_calls_without_a_final_state_are_v6_class_at_every_length() -> None:
    for chunks, H, n, sm_count in itertools.product(
        CHUNK_COUNTS, HEADS, loader.NUM_HOUSEHOLDER, SM_COUNTS
    ):
        assert (
            loader.prefill_body_class(chunks, H, n, sm_count, store_final_state=False)
            == V6
        ), (chunks, H, n, sm_count)


def test_short_sequences_are_v6_class() -> None:
    """Every chunk count below ``ROUTE_MIN_CHUNKS_P3`` (the empty sequence included) is V6-class, whatever the
    head count, the Householder count and the SM count."""
    for chunks in range(loader.ROUTE_MIN_CHUNKS_P3):
        for H, n, sm_count in itertools.product(
            HEADS, loader.NUM_HOUSEHOLDER, SM_COUNTS
        ):
            assert loader.prefill_body_class(chunks, H, n, sm_count) == V6, (
                chunks,
                H,
                n,
                sm_count,
            )


def test_merged_class_sequences_are_cut() -> None:
    """A merged-class sequence is cut by the split rule of record (``split_count`` gives ``p >= 2``): the merged
    class always runs a split program.  A merged-class sequence is also at least ``ROUTE_MIN_CHUNKS_P3`` chunks
    long, and some long sequence is merged-class."""
    merged = 0
    for chunks, H, n, sm_count in itertools.product(
        CHUNK_COUNTS, HEADS, loader.NUM_HOUSEHOLDER, SM_COUNTS
    ):
        if loader.prefill_body_class(chunks, H, n, sm_count) != P3:
            continue
        merged += 1
        assert chunks >= loader.ROUTE_MIN_CHUNKS_P3
        p, span = loader.split_count(chunks, H, n, sm_count)
        assert p >= 2 and 0 < span < chunks, (chunks, H, n, sm_count, p, span)
        cost_v6, cost_p3 = loader.prefill_body_costs(chunks, H, n, sm_count)
        assert cost_p3 < cost_v6
    assert merged > 0


@pytest.mark.parametrize("H", PERF_HEADS)
def test_class_of_a_sequence_is_independent_of_the_batch(H: int) -> None:
    """A sequence runs the same body alone and in any batch, in any position."""
    lens = (8192, 1000, 0, 8192, 32768, 4096, 1, 129)
    alone = [_classes((L,), H)[0] for L in lens]
    assert alone == [
        loader.prefill_body_class(loader.split_chunks(L, N_HH), H, N_HH, SM_COUNT)
        for L in lens
    ]
    assert _classes(lens, H) == alone
    for order in (
        tuple(reversed(range(len(lens)))),
        (4, 0, 7, 2, 5, 1, 6, 3),
        (3, 3, 0, 0, 2),
    ):
        assert _classes(tuple(lens[i] for i in order), H) == [alone[i] for i in order]


def test_route_chunk_steps_clamp_like_the_split_rules() -> None:
    """``route_chunk_steps`` reads ``cu_seqlens`` as the device does: the start clamped to ``[0, total_tokens]``,
    the end to ``[start, total_tokens]``, so a decreasing or negative pair is an empty sequence."""
    cu, n, total = [20, 300, -5, 10], 3, 180
    steps = loader.route_chunk_steps(cu, n, total_tokens=total)
    assert steps == [8, 0, 2]
    assert steps == [
        loader.split_chunks(end - start, n)
        for start, end in (
            loader.split_sequence_bounds(cu[i], cu[i + 1], total)
            for i in range(len(cu) - 1)
        )
    ]
    # the item table of the same batch reads the same lengths: one item per (sequence, head) under the exact mode
    table = loader.split_table(
        cu, n, 2, SM_COUNT, split_mode="exact", total_tokens=total
    )
    assert [span for seq, head, cut, span in table if head == 0] == steps
    mixed = loader.route_chunk_steps([0, 8192, 9192, 9192, 17384], n)
    assert mixed == [384, 48, 0, 384]
    assert loader.route_chunk_steps([0], n) == []


# ---------------------------------------------------------------------------
# The launch plan
# ---------------------------------------------------------------------------


def test_launch_plan() -> None:
    """No sequence -> no launch; a uniform batch -> one launch over every sequence; a mixed batch -> one launch per
    body in ``ROUTE_BODIES`` order (the merged body first), each over the ascending indices of its sequences."""
    assert loader.prefill_launch_plan([]) == []
    assert loader.prefill_launch_plan([V6] * 4) == [(V6, [0, 1, 2, 3])]
    assert loader.prefill_launch_plan([P3] * 2) == [(P3, [0, 1])]
    assert loader.prefill_launch_plan([P3, V6, V6, P3]) == [(P3, [0, 3]), (V6, [1, 2])]
    assert loader.prefill_launch_plan([V6, P3]) == [(P3, [1]), (V6, [0])]


def test_mixed_batch_of_record() -> None:
    """The batch (8192, 1000, 0, 8192) at n = 3 on 132 SMs: mixed at H = 10 (the long sequences merged-class, the
    short and the empty one V6-class), uniform V6 at H = 80, uniform merged with unit gates."""
    classes = _classes(MIXED_LENS, 10)
    assert classes == [P3, V6, V6, P3]
    assert loader.prefill_launch_plan(classes) == [(P3, [0, 3]), (V6, [1, 2])]
    assert _classes(MIXED_LENS, 80) == [V6] * 4
    assert _classes(MIXED_LENS, 10, unit_gates=True) == [P3] * 4
    assert _classes(MIXED_LENS, 10, unit_beta=True) == [P3] * 4
    assert _classes(MIXED_LENS, 10, store_final_state=False) == [V6] * 4
    # the compact offsets of the merged class are cut by the split rule: the mixed merged launch has a table
    merged = (0, MIXED_LENS[0], MIXED_LENS[0] + MIXED_LENS[3])
    table = loader.prefill_split_table(
        merged, num_heads=10, num_householder=N_HH, sm_count=SM_COUNT
    )
    assert table is not None and len(table) > 2 * 10
    assert {seq for seq, head, cut, span in table} == {0, 1}


# ---------------------------------------------------------------------------
# The batch body route: one body, one launch per sm_90a call
# ---------------------------------------------------------------------------

# The prefill bench workloads (H100, 132 SMs, n = 3, 2026-11-24): family A is d one-token sequences then one
# long sequence (``min(L, 65536 - d)`` tokens), B one 65536-token sequence, C the pair (36864, 28672).  Measured with
# each body forced on the whole batch, the merged split body is the faster kernel on every H10 / H20 row but C_H20
# (4000 us merged vs 3562 us V6); V6 is the faster on every H40 / H80 row.
BENCH_TOTAL = 65536
BENCH_LENGTHS = (8192, 16384, 32768, 65536)
BENCH_SHORT = (0, 1, 4, 8, 15)


def _bench_rows() -> list[tuple[str, int, tuple[int, ...]]]:
    rows = []
    for H in (10, 20, 40, 80):
        for L in BENCH_LENGTHS:
            for d in BENCH_SHORT:
                rows.append(
                    (f"A_H{H}_L{L}_d{d}", H, (1,) * d + (min(L, BENCH_TOTAL - d),))
                )
        rows.append((f"B_H{H}", H, (BENCH_TOTAL,)))
        rows.append((f"C_H{H}", H, (36864, 28672)))
    return rows


BENCH_ROWS = _bench_rows()


def _bench_body(name: str, H: int) -> str:
    return V6 if H >= 40 or name == "C_H20" else P3


@pytest.mark.parametrize("name,H,lens", BENCH_ROWS, ids=[row[0] for row in BENCH_ROWS])
def test_batch_body_of_the_bench_rows(name: str, H: int, lens: tuple[int, ...]) -> None:
    """The one body of each bench row is its measured faster whole-batch kernel; a mixed row (d > 0) takes the body
    of its d = 0 row, so the one-token sequences run in the long sequence's launch."""
    body = loader.prefill_batch_body(_offsets(lens), N_HH, H, SM_COUNT)
    assert body == _bench_body(name, H), (
        name,
        body,
        loader.prefill_batch_costs(_offsets(lens), N_HH, H, SM_COUNT),
    )


def test_pinned_batch_costs() -> None:
    """The modeled whole-batch kernel times in ns of one 65536-token sequence at H = 10 (the split rule's 13 items
    of 240 chunks per head, a warm-up of 8 for the cut ones, against one V6 tile of 3072 chunks per head; measured
    1905 us and 6222 us) and of the C pair at H = 20 (V6 runs the 1728-chunk tile; the merged body four items on a
    CTA)."""
    assert loader.prefill_batch_costs((0, 65536), N_HH, 10, SM_COUNT) == (
        loader.ROUTE_BATCH_V6_BASE_NS + 3072 * loader.ROUTE_BATCH_V6_STEP_NS,
        loader.ROUTE_BATCH_P3_BASE_NS + 248 * loader.ROUTE_BATCH_P3_STEP_NS,
    )
    t_v6, t_p3 = loader.prefill_batch_costs((0, 36864, 65536), N_HH, 20, SM_COUNT)
    assert t_v6 == loader.ROUTE_BATCH_V6_BASE_NS + 1728 * loader.ROUTE_BATCH_V6_STEP_NS
    assert t_v6 < t_p3
    assert loader.prefill_batch_costs((0,), N_HH, 10, SM_COUNT) == (0, 0)


@pytest.mark.parametrize("H", PERF_HEADS)
@pytest.mark.parametrize("N,L", PERF_NL)
def test_batch_body_of_the_performance_rows(H: int, N: int, L: int) -> None:
    """A uniform performance row keeps the body of its sequences' class."""
    body = P3 if (H, N, L) in MERGED_ROWS else V6
    assert loader.prefill_batch_body(_offsets((L,) * N), N_HH, H, SM_COUNT) == body


def test_batch_body_flags_and_candidacy() -> None:
    """A unit call is merged at every length, a call keeping no state V6 at every length, and a batch with no
    merged-class sequence is V6 whatever its size; the merged choice needs a merged-class sequence, which the split
    rule cuts."""
    for lens, H in ((MIXED_LENS, 10), ((1,) * 15 + (65521,), 20), ((300,) * 64, 10)):
        cu = _offsets(lens)
        assert loader.prefill_batch_body(cu, N_HH, H, SM_COUNT, unit_gates=True) == P3
        assert loader.prefill_batch_body(cu, N_HH, H, SM_COUNT, unit_beta=True) == P3
        assert (
            loader.prefill_batch_body(cu, N_HH, H, SM_COUNT, store_final_state=False)
            == V6
        )
    assert loader.prefill_batch_body(_offsets((300,) * 64), N_HH, 10, SM_COUNT) == V6
    assert loader.prefill_batch_body(_offsets((0, 1, 0)), N_HH, 10, SM_COUNT) == V6
    assert loader.prefill_batch_body(_offsets(MIXED_LENS), N_HH, 10, SM_COUNT) == P3
    for chunks, H, n, sm_count in itertools.product(
        (384, 768, 1536, 3072), HEADS, loader.NUM_HOUSEHOLDER, SM_COUNTS
    ):
        L = chunks * 64 // n
        for lens in ((L,), (1, 1, L), (L, L)):
            cu = _offsets(lens)
            if loader.prefill_batch_body(cu, n, H, sm_count) == P3:
                assert P3 in loader.prefill_body_classes(cu, n, H, sm_count)
                assert (
                    loader.prefill_split_table(
                        cu, num_heads=H, num_householder=n, sm_count=sm_count
                    )
                    is not None
                )


def test_one_token_sequences_cost_their_own_items() -> None:
    """In the merged body a one-token sequence is one item of two chunks per head (no cut, no warm-up): adding d of
    them to a merged long sequence moves its modeled time by at most the ``ceil(d H / SM count)`` such items a CTA
    takes."""
    for H in (10, 20):
        base = loader.prefill_batch_costs((0, 65536), N_HH, H, SM_COUNT)[1]
        for d in BENCH_SHORT:
            lens = (1,) * d + (65536 - d,)
            assert {
                (span, cut)
                for seq, head, cut, span in loader.split_table(
                    _offsets(lens), N_HH, H, SM_COUNT
                )
                if seq < d
            } <= {(2, 0)}
            t_p3 = loader.prefill_batch_costs(_offsets(lens), N_HH, H, SM_COUNT)[1]
            per_cta = -(-(d * H) // SM_COUNT)
            assert t_p3 - base <= per_cta * 2 * loader.ROUTE_BATCH_P3_STEP_NS, (
                H,
                d,
                t_p3 - base,
            )


# ---------------------------------------------------------------------------
# The V6 warm-up split plan
# ---------------------------------------------------------------------------

# The balanced rows of the bench rows' long sequences on 132 SMs (H -> rows per long length; "C" the pair): a
# bench row has these rows plus one whole-head row per head of each one-token sequence.
BENCH_BALANCED_ROWS = {
    10: {8192: 130, 16384: 134, 32768: 140, 65536: 141, "C": 280},
    20: {8192: 140, 16384: 148, 32768: 150, 65536: 151, "C": 298},
    40: {8192: 163, 16384: 167, 32768: 169, 65536: 171, "C": 339},
    80: {8192: 203, 16384: 207, 32768: 210, 65536: 210, "C": 417},
}


def _v6_plan(lens: tuple[int, ...], H: int, n: int = N_HH, sm_count: int = SM_COUNT):
    return loader.prefill_v6_sm90_plan(
        _offsets(lens), num_heads=H, num_householder=n, sm_count=sm_count
    )


def _covers_every_chunk_once(lens, H, n, plan) -> None:
    """The rows' head-major write ranges tile every (sequence, head)'s chunks exactly once: a balanced row writes
    from its cut to the next cut of its (sequence, head) (or the end), a whole-head row its whole head."""
    chunks = loader.v6_sm90_chunks(_offsets(lens), n)
    cuts: dict[tuple[int, int], list[int]] = {}
    for seq, head, cut, span in plan.rows:
        cuts.setdefault((seq, head), []).append(cut)
        if span >= 0:
            assert cut == 0 and span == chunks[seq], (seq, head, cut, span)
    assert set(cuts) == {(q, h) for q in range(len(lens)) for h in range(H)}
    for (seq, head), at in cuts.items():
        assert sorted(at) == sorted(set(at)) and min(at) == 0, (seq, head, at)
        assert all(cut < max(chunks[seq], 1) for cut in at), (seq, head, at)


@pytest.mark.parametrize("name,H,lens", BENCH_ROWS, ids=[row[0] for row in BENCH_ROWS])
def test_v6_split_plan_of_the_bench_rows(
    name: str, H: int, lens: tuple[int, ...]
) -> None:
    """Every bench row runs the V6 split program on 132 SMs: its long sequences balanced (the pinned row counts),
    each one-token sequence one whole-head row per head, every CTA's list within the plan table, the rows
    covering every chunk once, the plan lighter than the V6 program's tiles."""
    plan = _v6_plan(lens, H)
    assert plan.split and plan.grid == SM_COUNT, (name, plan.load, plan.serial_load)
    assert plan.load < plan.serial_load
    d = lens.count(1)
    if name.startswith("C_"):
        key = "C"
    else:  # the long sequence of an A row (65536 - d tokens at L = 65536: the same chunks) or of a B row
        key = min(BENCH_LENGTHS, key=lambda L: abs(L - max(lens)))
    assert len(plan.rows) == BENCH_BALANCED_ROWS[H][key] + d * H, (name, len(plan.rows))
    assert len(plan.offsets) == plan.grid + 1 and plan.offsets[-1] == len(plan.rows)
    for b in range(plan.grid):
        here = plan.rows[plan.offsets[b] : plan.offsets[b + 1]]
        balanced = [row for row in here if row[3] < 0]
        assert here[: len(balanced)] == tuple(balanced)  # the balanced rows lead
        assert len(balanced) <= loader.V6_SM90_PLAN_ROWS
    _covers_every_chunk_once(lens, H, N_HH, plan)


def test_v6_split_long_rows_do_not_depend_on_the_batch() -> None:
    """A balanced sequence's rows, and the CTA of each, are the same with any one-token or short sequences beside
    it (the long sequence's output does not depend on them)."""
    for H in (10, 40, 80):
        alone = _v6_plan((65536,), H)
        mine = [
            (b, row[1:])
            for b in range(alone.grid)
            for row in alone.rows[alone.offsets[b] : alone.offsets[b + 1]]
        ]
        for lens in ((1,) * 15 + (65536,), (1, 300, 0, 65536)):
            plan = _v6_plan(lens, H)
            seq = len(lens) - 1
            theirs = [
                (b, row[1:])
                for b in range(plan.grid)
                for row in plan.rows[plan.offsets[b] : plan.offsets[b + 1]]
                if row[0] == seq
            ]
            assert theirs == mine, (H, lens)


@pytest.mark.parametrize(
    "lens,H",
    (
        ((1024,) * 64, 10),
        ((256,) * 64, 80),
        ((1024,) * 16, 80),
        ((1024,) * 16, 24),
        ((300,) * 64, 10),
        ((4096,) * 4, 10),
        ((8192,), 3),
        ((8192,), 1),
        ((1,) * 64, 80),
        ((0, 0), 10),
    ),
)
def test_v6_split_is_not_taken_where_the_tiles_fill_the_sms(lens, H) -> None:
    """Short-only batches, and batches whose (sequence, head) tiles already fill the SMs, keep the V6 program: no
    sequence is balanced, or the plan's most loaded CTA is not lighter than the V6 program's (e.g. 64 x 1024 at
    H = 10, 16 x 1024 at H = 80)."""
    plan = _v6_plan(lens, H)
    assert not plan.split, (lens, H, plan.load, plan.serial_load)


def test_v6_split_item_words() -> None:
    """The item words: four per row, the ``grid + 1`` offsets, then ``N`` and the planned ``cu_seqlens``."""
    lens = (1, 0, 8192, 65)
    plan = _v6_plan(lens, 10)
    words = loader.v6_sm90_item_words(plan, _offsets(lens))
    rows = len(plan.rows)
    assert words[: 4 * rows] == tuple(w for row in plan.rows for w in row)
    assert words[4 * rows : 4 * rows + plan.grid + 1] == plan.offsets
    assert words[4 * rows + plan.grid + 1 :] == (len(lens), *_offsets(lens))


def test_v6_split_plan_rules() -> None:
    """The slot rule (at least V6_SM90_MIN_SLOT chunks, shorter than a head, a multiple of n), the cut-min rule (no
    cut fewer than V6_SM90_CUT_MIN chunks after its head start), cut-0 rows first in every CTA list, at most
    V6_SM90_PLAN_ROWS // 2 balanced sequences (the longest), and the serial load of the V6 tiles."""
    for H, n, L in itertools.product(
        (8, 10, 24, 40, 80), (1, 2, 3, 4), (6000, 8192, 32768)
    ):
        plan = _v6_plan((L,), H, n)
        (chunks,) = loader.v6_sm90_chunks(_offsets((L,)), n)
        slot = loader.v6_sm90_slot(chunks, H, n, SM_COUNT)
        if not slot:
            continue
        assert loader.V6_SM90_MIN_SLOT <= slot < chunks + n and slot % n == 0
        for b in range(plan.grid):
            here = plan.rows[plan.offsets[b] : plan.offsets[b + 1]]
            first = [r[2] == 0 for r in here]
            assert first == sorted(first, reverse=True)
        for _seq, _head, cut, span in plan.rows:
            assert span == -slot and (cut == 0 or cut >= loader.V6_SM90_CUT_MIN)
        _covers_every_chunk_once((L,), H, n, plan)
    lens = (8192,) * 40
    plan = _v6_plan(lens, 80)
    balanced = {row[0] for row in plan.rows if row[3] < 0}
    assert len(balanced) <= loader.V6_SM90_PLAN_ROWS // 2
    assert (
        loader.v6_sm90_serial_load([384, 1], 10, SM_COUNT)
        == 384 + loader.V6_SM90_ROW_COST
    )
    assert loader.v6_sm90_serial_load([], 10, SM_COUNT) == 0


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


def test_sm90_prefill_route_lists_the_eleven_programs() -> None:
    route = loader.ROUTES[ARCH][loader.PREFILL_ROUTE]
    assert route == [*V6_NAMES, *UNIT_NAMES, *SPLIT_NAMES, *V6_SPLIT_NAMES]
    assert loader.prefill_route_modules(ARCH) == tuple(route)
    for name in V6_NAMES:
        assert loader.MODULES[name]["body"] == V6
        assert not loader.MODULES[name]["build"].get(loader.PREFILL_SPLIT_FLAG)
    for name in UNIT_NAMES:
        record = loader.MODULES[name]
        assert record["body"] == P3
        assert record["build"][loader.PREFILL_SPLIT_FLAG] is False
        assert record["build"]["unit_gates"] != record["build"]["unit_beta"]
    for name in SPLIT_NAMES:
        record = loader.MODULES[name]
        assert record["body"] == P3
        assert record["build"][loader.PREFILL_SPLIT_FLAG] is True
    for name in V6_SPLIT_NAMES:
        record = loader.MODULES[name]
        assert record["body"] == V6
        assert record["build"][loader.PREFILL_SPLIT_FLAG] is True
        assert record["build"]["csr"] is True and record["build"]["plan_guard"] is True
    # no other record of the prefill role is delivered for sm_90a
    assert {
        name
        for name, record in loader.MODULES.items()
        if record["role"] == "prefill" and ARCH in record["arches"]
    } == set(route)


def test_selected_prefill_module_filters_by_body() -> None:
    """The body filter resolves the three V6 flag sets (exact and split), the two unit sets and the three merged split
    sets to their records and refuses the removed merged exact programs with ``0 programs``; no body searches every
    body."""
    for flags, name in zip(V6_FLAGS, V6_NAMES, strict=True):
        assert _select(V6, flags) == name
    for flags, name in zip(UNIT_FLAGS, UNIT_NAMES, strict=True):
        assert _select(P3, flags) == name
    for flags, name in zip(SPLIT_FLAGS, SPLIT_NAMES, strict=True):
        assert _select(P3, flags, **{loader.PREFILL_SPLIT_FLAG: True}) == name
    for flags, name in zip(V6_FLAGS, V6_SPLIT_NAMES, strict=True):
        assert _select(V6, flags, **{loader.PREFILL_SPLIT_FLAG: True}) == name
    for flags in REMOVED_FLAGS:
        with pytest.raises(ValueError, match="0 programs of the 'prefill' route"):
            _select(P3, flags)
    # no body (None) searches every body of the route: the plain flags resolve to the V6 plain program alone
    assert (
        loader.selected_prefill_module(
            ARCH,
            use_initial_state=False,
            use_state_indices=False,
            store_final_state=True,
        )
        == V6_NAMES[0]
    )
    for flags in (*UNIT_FLAGS, (False, False, False, False, False)):
        with pytest.raises(ValueError, match="0 programs of the 'prefill' route"):
            _select(V6, flags)
        with pytest.raises(ValueError, match="0 programs of the 'prefill' route"):
            _select(V6, flags, **{loader.PREFILL_SPLIT_FLAG: True})


def test_v6_record_names_are_the_kernel_digests() -> None:
    """A V6 record is named ``cake_gdp_<20 hex>``, the digest its kernel was rendered under; the record's kernel,
    library and symbol carry that name, the three names differ, and the build manifest lists each library for
    sm_90a."""
    libraries = cake_build.libraries_by_name()
    names = (*V6_NAMES, *V6_SPLIT_NAMES)
    assert len(set(names)) == len(names)
    for name in names:
        record = loader.MODULES[name]
        assert re.fullmatch(r"cake_gdp_[0-9a-f]{20}", name), name
        assert record["kernel"] == record["library"] == name
        assert record["kernel_symbol"] == f"kernel_{name}"
        assert f"{name}_sm_90a" in libraries, name


def test_v6_records_share_the_merged_launch_contract() -> None:
    """A V6 record launches 384 threads per CTA under the ``tiles`` grid rule and takes the merged body's 23
    formals in the merged body's order, so the backend binds one ``values`` dict for every launch."""
    merged = loader.MODULES[UNIT_NAMES[0]]
    for name in (*V6_NAMES, *V6_SPLIT_NAMES):
        split = name in V6_SPLIT_NAMES
        record = loader.MODULES[name]
        assert record["body"] == V6 and record["arches"] == [ARCH]
        assert record["launch"]["block"] == (384, 1, 1)
        assert record["grid_rule"] == {"kind": "tiles"}
        assert record["arg_plan"] == merged["arg_plan"]
        assert record["logical"] == merged["logical"]
        assert (
            set(record["logical"].values())
            == (
                set(loader.PREFILL_LOGICAL_TENSORS)
                | set(loader.PREFILL_LOGICAL_SCALARS)
            )
            - BLACKWELL_ONLY_LOGICAL
        )
        assert record["select_rule"]["kind"] == "prefill_flags"
        assert set(record["select_rule"]) == {
            "kind",
            *loader.PREFILL_FLAGS,
            *((loader.PREFILL_SPLIT_FLAG,) if split else ()),
        }
        assert record["select_rule"]["store_final_state"] is True
        assert {flag: record["build"][flag] for flag in loader.PREFILL_FLAGS} == {
            flag: record["select_rule"][flag] for flag in loader.PREFILL_FLAGS
        }
    assert merged["launch"]["block"] == (512, 1, 1)


def test_body_route_block_sits_between_the_split_rules_and_the_table() -> None:
    """The BODY ROUTE OF RECORD block of ``cake_jit.py`` (one text in every copy) follows the END SPLIT RULES marker
    and precedes ``prefill_split_table``."""
    source = inspect.getsource(loader)
    end_split = source.index("# ---- END SPLIT RULES OF RECORD ----")
    start = source.index("# ---- BODY ROUTE OF RECORD")
    end = source.index("# ---- END BODY ROUTE OF RECORD ----")
    table = source.index("def prefill_split_table(")
    assert end_split < start < end < table
    block = source[start:end]
    for symbol in (
        "ROUTE_BODY_P3",
        "ROUTE_BODY_V6",
        "ROUTE_BODIES",
        "ROUTE_MIN_CHUNKS_P3",
        "def route_chunk_steps(",
        "def prefill_body_costs(",
        "def prefill_body_class(",
        "def prefill_body_classes(",
        "def prefill_launch_plan(",
    ):
        assert symbol in block, symbol
    for symbol in (
        "ROUTE_BODIES",
        "ROUTE_BODY_P3",
        "ROUTE_BODY_V6",
        "prefill_body_class",
        "prefill_body_classes",
        "prefill_body_costs",
        "prefill_launch_plan",
        "route_chunk_steps",
    ):
        assert symbol in loader.__all__, symbol
