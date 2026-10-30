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

"""Plain work bins of short bounded direct M128 N32 launches and the composed
work-bin decision of a direct launch (CPU; no GPU needed).

A bounded (lower-bound gate) N32 launch that the strict bin test
(``_use_direct_work_bins``) keeps one task per CTA asks for its module's plain
work-binned build where, with more tasks than SMs and a longest sequence of at
most ``BF16_BOUNDED_WORK_BIN_MAX_TASK_CHUNKS`` chunks, the bins that charge
every task ``BF16_BOUNDED_WORK_BIN_PROBE_TASK_COST`` chunks have a largest bin
at least ``BF16_BOUNDED_WORK_BIN_MIN_MARGIN`` chunks above the longest task plus
that cost (``_use_bounded_work_bins``, ``_bounded_bins_route``).  It never takes
the turnover build, and its bin table charges
``BF16_BOUNDED_WORK_BIN_TASK_COST`` chunks per task.  ``_direct_work_bin_route``
composes this rule with the strict test, the bounded turnover predicate of
strict launches and the turnover bins of unbounded launches; each launch takes
at most one of them.
"""

import itertools

import pytest

import flashinfer.cake_kda_tf32_runtime as runtime

CHUNK = runtime.BF16_M128_CHUNK
PROBE = runtime.BF16_BOUNDED_WORK_BIN_PROBE_TASK_COST
MARGIN = runtime.BF16_BOUNDED_WORK_BIN_MIN_MARGIN
LIMIT = runtime.BF16_BOUNDED_WORK_BIN_MAX_TASK_CHUNKS
COST = runtime.BF16_BOUNDED_WORK_BIN_TASK_COST
DIRECT = runtime.BF16_ROUTE_DIRECT_M128
N16 = runtime.BF16_ROUTE_DIRECT_M128_N16
BOUNDED = runtime.KDAGateKind.LOWER_BOUND
UNBOUNDED = runtime.KDAGateKind.UNBOUNDED_SOFTPLUS
PARTS = [
    pytest.param("sm_100a", 148, id="sm_100a-148"),
    pytest.param("sm_103a", 152, id="sm_103a-152"),
]
MIXED = (17, 64, 65, 127, 128, 200, 255, 96) * 2

# (lengths, heads, margin on sm_100a-148, on sm_103a-152): the margin is the
# largest bin of the table that charges every task the probe cost, less the
# longest task plus that cost.  Every row keeps one task per CTA under the
# strict test on both parts and has more tasks than SMs.
MARGIN_SHAPES = [
    # 128 long tasks fill one bin each; the 3-chunk tasks fill the other bins
    # and stack two to a bin.
    pytest.param((256,) * 8 + (96,) * 2, 16, 3, 3, id="8x256+2x96-h16"),
    # One 3-chunk and one 2-chunk sequence: two short tasks share a bin.
    pytest.param((256,) * 8 + (96, 64), 16, 2, 1, id="8x256+96+64-h16"),
    pytest.param((256,) * 8 + (96, 64, 64), 16, 7, 2, id="8x256+96+2x64-h16"),
    pytest.param((256,) * 8 + (128,) * 2, 16, 5, 5, id="8x256+2x128-h16"),
    # The short mixed rows: H16 above the margin, H12 below it.
    pytest.param(MIXED, 16, 5, 5, id="mixed-h16"),
    pytest.param(MIXED, 12, 2, 2, id="mixed-h12"),
]


def _strict(lengths, heads, sm_count, chunk_tokens=CHUNK):
    return runtime._use_direct_work_bins(
        tuple(lengths), num_heads=heads, sm_count=sm_count, chunk_tokens=chunk_tokens
    )


def _margin(lengths, heads, sm_count):
    """Largest bin of the probe-cost table less the longest task plus the cost."""
    table, num_bins = runtime._make_direct_work_bins(
        tuple(lengths),
        num_heads=heads,
        worker_count=sm_count,
        chunk_tokens=CHUNK,
        task_cost=PROBE,
    )
    chunks = [(n + CHUNK - 1) // CHUNK for n in lengths]
    largest = max(
        sum(chunks[task // heads] + PROBE for task in table[table[i] : table[i + 1]])
        for i in range(num_bins)
    )
    return largest - (max(chunks) + PROBE)


def _bounded_bins(lengths, heads, sm_count):
    return runtime._use_bounded_work_bins(
        tuple(lengths), num_heads=heads, sm_count=sm_count
    )


def _route(lengths, heads, arch, sm_count, **change):
    facts = dict(
        route=DIRECT,
        tf32_direct=False,
        n32_ft_slab=False,
        gate_kind=BOUNDED,
        seq_lens=tuple(lengths),
        max_seq_len=max(lengths),
        num_heads=heads,
        sm_count=sm_count,
        gpu_arch=arch,
    )
    facts.update(change)
    return runtime._direct_work_bin_route(**facts)


@pytest.mark.cpu_only
@pytest.mark.parametrize(("arch", "sm_count"), PARTS)
@pytest.mark.parametrize(("lengths", "heads", "on_b200", "on_gb300"), MARGIN_SHAPES)
def test_bounded_bins_take_launches_from_the_margin_on(
    arch, sm_count, lengths, heads, on_b200, on_gb300
):
    """Both sides of the margin on each part: the rule takes a launch exactly
    where the probe-cost table's margin reaches ``BF16_BOUNDED_WORK_BIN_MIN_MARGIN``."""
    margin = on_b200 if sm_count == 148 else on_gb300
    assert len(lengths) * heads > sm_count
    assert not _strict(lengths, heads, sm_count)
    assert max(lengths) <= CHUNK * LIMIT
    assert _margin(lengths, heads, sm_count) == margin
    assert _bounded_bins(lengths, heads, sm_count) == (margin >= MARGIN)


@pytest.mark.cpu_only
def test_bounded_bin_constants():
    assert (LIMIT, PROBE, MARGIN, COST) == (8, 5, 3, 3)


@pytest.mark.cpu_only
@pytest.mark.parametrize(("arch", "sm_count"), PARTS)
def test_bounded_bins_stay_within_the_chunk_limit(arch, sm_count):
    """A longest task of 8 chunks takes the bins; one of 9 chunks does not,
    although its margin also reaches the minimum (both parts)."""
    within = (CHUNK * LIMIT,) * 8 + (128,) * 2
    past = (CHUNK * LIMIT + 1,) * 8 + (128,) * 2
    for lengths in (within, past):
        assert not _strict(lengths, 16, sm_count)
        assert _margin(lengths, 16, sm_count) >= MARGIN
    assert _bounded_bins(within, 16, sm_count)
    assert not _bounded_bins(past, 16, sm_count)


@pytest.mark.cpu_only
@pytest.mark.parametrize(("arch", "sm_count"), PARTS)
def test_bounded_bins_need_more_tasks_than_sms(arch, sm_count):
    """One 8-chunk task and 3-chunk tasks, one head: with as many tasks as SMs
    each task has its own bin (margin 0); one task more stacks two short tasks
    in a bin (margin 3)."""
    at = (CHUNK * LIMIT,) + (96,) * (sm_count - 1)
    above = (CHUNK * LIMIT,) + (96,) * sm_count
    assert len(at) == sm_count and len(above) == sm_count + 1
    assert not _strict(at, 1, sm_count) and not _strict(above, 1, sm_count)
    assert _margin(at, 1, sm_count) == 0
    assert _margin(above, 1, sm_count) == 3
    assert not _bounded_bins(at, 1, sm_count)
    assert _bounded_bins(above, 1, sm_count)
    assert not _bounded_bins((), 16, sm_count)
    assert not _bounded_bins(MIXED[:-1] + (0,), 16, sm_count)


def _bins_route_facts(lengths, heads, sm_count):
    return dict(
        strict=_strict(lengths, heads, sm_count),
        route=DIRECT,
        tf32_direct=False,
        n32_ft_slab=False,
        seq_lens=tuple(lengths),
        num_heads=heads,
        sm_count=sm_count,
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(("arch", "sm_count"), PARTS)
@pytest.mark.parametrize(("lengths", "heads", "on_b200", "on_gb300"), MARGIN_SHAPES)
def test_bounded_bins_route_takes_bounded_n32_launches_only(
    arch, sm_count, lengths, heads, on_b200, on_gb300
):
    """``_bounded_bins_route``: a bounded direct N32 launch takes the bins
    exactly where the rule does; an unbounded launch never does, nor a strict
    work-binned, TF32 direct, N32 FT slab or N16 launch."""
    want = (on_b200 if sm_count == 148 else on_gb300) >= MARGIN
    facts = _bins_route_facts(lengths, heads, sm_count)
    assert runtime._bounded_bins_route(**facts, bounded=True) == want
    assert not runtime._bounded_bins_route(**facts, bounded=False)
    for change in (
        dict(strict=True),
        dict(tf32_direct=True),
        dict(n32_ft_slab=True),
        dict(route=N16),
    ):
        assert not runtime._bounded_bins_route(**{**facts, **change}, bounded=True), (
            change
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize(("arch", "sm_count"), PARTS)
def test_bounded_bins_launch_builds_the_plain_bins_with_the_task_cost(arch, sm_count):
    """A launch the rule takes asks for work bins charged
    ``BF16_BOUNDED_WORK_BIN_TASK_COST`` and never for the turnover build, also
    where the turnover route and the bounded turnover predicate would admit it;
    off the rule's domain the launch keeps one task per CTA."""
    for lengths, heads in (
        (MIXED, 16),
        ((256,) * 8 + (96,) * 2, 16),
        ((256,) * 2 + (128,) * 12, 16),
        ((256,) * 3 + (128,) * 14, 12),
        ((256,) * 6 + (128,) * 4, 16),
    ):
        assert _bounded_bins(lengths, heads, sm_count), lengths
        got = _route(lengths, heads, arch, sm_count)
        assert got == runtime._DirectWorkBinRoute(
            work_bins=True, turnover_bins_only=False, turnover=False, task_cost=COST
        ), lengths
        unbounded = _route(lengths, heads, arch, sm_count, gate_kind=UNBOUNDED)
        assert unbounded.task_cost in (0, runtime.BF16_WORK_BIN_TURNOVER_TASK_COST)
        assert unbounded.turnover_bins_only == unbounded.work_bins
    # The C8 rows: the turnover route and the bounded turnover predicate (on
    # the strict table) both admit them; the bounded rule keeps them off it.
    for lengths, heads in (
        ((256,) * 2 + (128,) * 12, 16),
        ((256,) * 3 + (128,) * 14, 12),
        ((256,) * 6 + (128,) * 4, 16),
    ):
        assert runtime._work_bin_turnover_route(
            route_work_bins=True,
            direct_n16=False,
            n32_ft_slab=False,
            max_seq_len=max(lengths),
            gpu_arch=arch,
            sm_count=sm_count,
        )
        assert runtime._bounded_turnover_route(
            bounded_gate=True,
            seq_lens=tuple(lengths),
            num_heads=heads,
            gpu_arch=arch,
            sm_count=sm_count,
        )
    for lengths in (MIXED, (256,) * 8 + (96, 64)):
        assert _route(lengths, 12, arch, sm_count) == runtime._DirectWorkBinRoute()


# Shapes of the composed-decision sweep: the margin rows, the chunk-limit and
# task-count rows, the turnover-bin rows, strict work-binned rows (bounded
# turnover and work-binned builds) and long rows past every turnover limit.
SWEEP_SHAPES = [
    *((p.values[0], p.values[1]) for p in MARGIN_SHAPES),
    ((256,) * 8 + (288,), 16),
    ((288,) * 8 + (128,) * 2, 16),
    ((256,) + (96,) * 147, 1),
    ((256,) + (96,) * 148, 1),
    ((256,) + (96,) * 151, 1),
    ((256,) + (96,) * 152, 1),
    ((256, 17) * 8, 16),
    ((256, 17) * 8, 12),
    ((256,) * 4 + (32,) * 12, 16),
    ((80, 149, 253, 29, 43, 155, 81, 202, 253, 43), 16),
    ((58, 45, 226, 189, 72, 106, 199, 40, 142, 142, 57, 119), 16),
    ((77, 181, 220, 207, 22, 110, 250, 52, 73, 120, 111, 117, 103), 12),
    ((77, 181, 220, 207, 22, 110, 250, 52, 73, 120, 111, 117, 103), 16),
    ((22, 18, 16, 51, 8, 15, 57, 10, 37, 62, 33, 29, 43, 47, 8, 39), 12),
    ((244, 179, 352, 463, 57, 193, 388, 229, 473, 50, 360, 364, 34, 372), 16),
    ((512, 16) * 8, 16),
    ((256,) * 2 + (128,) * 12, 16),
    ((256,) * 3 + (128,) * 14, 12),
    ((256,) * 6 + (128,) * 4, 16),
    ((64,) * 16, 12),
    ((64,) * 16, 16),
    ((128,) * 16, 16),
    ((256,) * 16, 12),
    ((64,) * 64, 16),
    ((96,) * 48, 16),
    ((64,) * 28 + (256,) * 4, 16),
    ((128, 160, 192, 100) * 12, 16),
    ((256,) * 11 + (64,) * 5, 16),
    ((1, 33, 64, 96) * 12, 16),
    ((1024, 64) * 8, 16),
    ((8192,), 12),
    ((4096,) * 4, 16),
]
VARIANTS = [
    dict(route=route, tf32_direct=tf32, n32_ft_slab=slab, gate_kind=gate)
    for route, tf32, slab, gate in itertools.product(
        (DIRECT, N16), (False, True), (False, True), (BOUNDED, UNBOUNDED)
    )
]


@pytest.mark.cpu_only
@pytest.mark.parametrize(("arch", "sm_count"), PARTS)
def test_composed_decision_claims_each_launch_once(arch, sm_count, monkeypatch):
    """Over the sweep, both gates, both direct bodies and the TF32 direct and
    N32 FT slab forms: no launch takes two rules (the bounded turnover of a
    strict launch, the turnover bins of an unbounded launch, the plain bins of
    a bounded launch), and the bounded rule changes exactly the launches it
    takes, each from one task per CTA to plain bins charged its task cost;
    every other launch (the strict, bounded-turnover and turnover-bin ones
    among them) routes as without the rule."""
    served = {"strict_turnover": 0, "turnover_bins": 0, "bounded_bins": 0}
    with_rule = {}
    for (lengths, heads), variant in itertools.product(SWEEP_SHAPES, VARIANTS):
        got = _route(lengths, heads, arch, sm_count, **variant)
        with_rule[(lengths, heads, tuple(variant.items()))] = got
        bounded = variant["gate_kind"] == BOUNDED
        n16 = variant["route"] == N16
        strict = (not variant["tf32_direct"]) and _strict(
            lengths,
            heads,
            sm_count,
            runtime.BF16_N16_M128_CHUNK if n16 else CHUNK,
        )
        bounded_bins = runtime._bounded_bins_route(
            strict=strict,
            route=variant["route"],
            tf32_direct=variant["tf32_direct"],
            n32_ft_slab=variant["n32_ft_slab"],
            bounded=bounded,
            seq_lens=tuple(lengths),
            num_heads=heads,
            sm_count=sm_count,
        )
        claims = [
            bounded and strict and got.turnover,
            got.turnover_bins_only,
            bounded_bins,
        ]
        assert sum(claims) <= 1, (lengths, heads, variant, got)
        for name, claim in zip(served, claims, strict=True):
            served[name] += claim
        if bounded_bins:
            assert got == runtime._DirectWorkBinRoute(work_bins=True, task_cost=COST), (
                lengths,
                heads,
                variant,
            )
    # The sweep reaches every rule on this part.
    assert all(served.values()), served
    monkeypatch.setattr(runtime, "_use_bounded_work_bins", lambda *a, **k: False)
    moved = 0
    for (lengths, heads, variant), got in with_rule.items():
        without = _route(lengths, heads, arch, sm_count, **dict(variant))
        if got != without:
            moved += 1
            assert without == runtime._DirectWorkBinRoute(), (lengths, heads)
            assert got == runtime._DirectWorkBinRoute(work_bins=True, task_cost=COST)
            assert dict(variant) == dict(
                route=DIRECT, tf32_direct=False, n32_ft_slab=False, gate_kind=BOUNDED
            )
    assert moved == served["bounded_bins"] > 0


@pytest.mark.cpu_only
@pytest.mark.parametrize(("arch", "sm_count"), PARTS)
def test_strict_and_turnover_bin_launches_keep_their_route(arch, sm_count):
    """The launches the other rules serve: a strict bounded launch reads the
    strict table (task cost 0) and takes the turnover build exactly where the
    bounded predicate holds; an unbounded launch the strict test keeps one task
    per CTA takes the turnover bins exactly where ``_use_turnover_work_bins``
    does."""
    for lengths, heads in (
        ((128,) * 16, 16),
        ((64,) * 16, 16),
        ((256,) * 16, 12),
        ((128, 160, 192, 100) * 12, 16),
    ):
        assert _strict(lengths, heads, sm_count)
        got = _route(lengths, heads, arch, sm_count)
        assert got.work_bins and not got.turnover_bins_only and got.task_cost == 0
        assert got.turnover == runtime._bounded_turnover_route(
            bounded_gate=True,
            seq_lens=tuple(lengths),
            num_heads=heads,
            gpu_arch=arch,
            sm_count=sm_count,
        )
    for lengths, heads in (
        (MIXED, 16),
        ((256, 17) * 8, 16),
        ((256,) * 8 + (96,) * 2, 16),
    ):
        assert not _strict(lengths, heads, sm_count)
        turn = runtime._use_turnover_work_bins(
            tuple(lengths), num_heads=heads, sm_count=sm_count, gpu_arch=arch
        )
        got = _route(lengths, heads, arch, sm_count, gate_kind=UNBOUNDED)
        assert got == (
            runtime._DirectWorkBinRoute(
                work_bins=True,
                turnover_bins_only=True,
                turnover=True,
                task_cost=runtime.BF16_WORK_BIN_TURNOVER_TASK_COST,
            )
            if turn
            else runtime._DirectWorkBinRoute()
        ), (lengths, heads)
