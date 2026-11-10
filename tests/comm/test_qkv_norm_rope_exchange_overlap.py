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

Tests of the fused QKV projection + norm + RoPE + head exchange.

Host-only tests check the tile plan (every destination element of a rank's
slab is written exactly once; readiness tickets add up), the plan policy, the
request validation, the launch-path resolution (``fold_barrier``), the row-count
gates (``shape_gates``), the per-(architecture, world size) group scaling
(``group_m_scale``) and producer build (``producer_variant``) defaults with
their environment overrides, and the loader tables.  The multi-GPU tests (two,
four or eight devices of one supported architecture) check both layouts
against an fp32 step-by-step reference including a tail row count, CUDA-graph
capture and replay with fresh inputs and eager calls in between, two sessions
in flight on one group, the refusals that protect a captured graph, the NCCL
path (the per-operator path: cuBLAS projection, ``rmsnorm``, RoPE) against a
reference that mirrors its own rounding points (the ranks agree on the
tolerance verdict before raising), the two launch paths against each other
(the default folded path two launches per call, the explicit barrier path
three), the large-row gates against the persistent launch, the baseline
producer build and group scaling (the two environment opt-outs) against the
defaults, and -- where the cross-destination-band build is delivered -- the
cross-destination-band plan against the band plan.  Every comparison of a
device output with a host reference is ``_check``: finite, then
``assert_close`` at ``atol = 0.05``, ``rtol = 1e-2``.  Seven comparisons are
bitwise (``torch.equal``): the replayed outputs of the two captured calls
against each other (part 2), the kept slot views of the NCCL path (part 4),
the folded against the barrier launch path (part 5), the gated against the
persistent launch (part 7), the opt-outs against the defaults (part 8), the
cross-destination-band against the band plan (part 9) and, host-only, the
``v`` of the two references.
"""

from __future__ import annotations

import math
import os
import random

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from flashinfer.comm import cake_qkv_exchange_plan as plans
from flashinfer.comm import cake_qkv_norm_rope_exchange_overlap as backend
from flashinfer.comm.qkv_norm_rope_exchange_overlap import validate_request
from flashinfer.jit import cake_qkv_norm_rope_exchange_overlap as loader

HEAD_DIM = 128
EPS = 1e-6
WORKER_TIMEOUT_SECONDS = 1800
# Head count of the negative case (part 6 of the multi-GPU test): 27 is not divisible by any supported world size
# (2, 4, 8), so every world size refuses it.
INDIVISIBLE_HEADS = 27


# --------------------------------------------------------------------------- host-only


def _destination_writes(plan: plans.TilePlan) -> dict[int, torch.Tensor]:
    """Per destination, how often each element of this source rank's slab is written by the tile table."""

    counts = {
        dest: torch.zeros(plan.slot_elems, dtype=torch.int64)
        for dest in range(plan.world)
    }
    columns = torch.arange(plan.bn)
    column_offset = (columns // HEAD_DIM) * plan.head_stride + columns % HEAD_DIM
    for row in plan.tiles.tolist():
        mt, _n0, dest, _kind, base_elem, row_stride, _word, _ticket = row
        if dest < 0:
            continue
        m = torch.arange(mt * plans.BLOCK_M, min(plan.M, (mt + 1) * plans.BLOCK_M))
        b, s = m // plan.s_local, m % plan.s_local
        row_offset = base_elem + b * plan.batch_stride + s * row_stride
        index = (row_offset[:, None] + column_offset[None, :]).flatten()
        counts[dest] += torch.bincount(index, minlength=plan.slot_elems)
    return counts


@pytest.mark.parametrize("layout", plans.LAYOUTS)
@pytest.mark.parametrize(
    "world_size, batch, s_local, hidden",
    [(2, 1, 128, 512), (2, 2, 100, 512), (4, 1, 200, 1024), (8, 1, 130, 1024)],
)
def test_plan_writes_every_destination_element_once(
    layout, world_size, batch, s_local, hidden
):
    geometry = plans.Geometry(
        world_size=world_size, batch=batch, s_local=s_local, hidden=hidden
    )
    # Every delivered architecture; SM107 has no 8-rank sessions (4-GPU nodes),
    # so that geometry is skipped there instead of asserted.
    for arch in sorted(set(loader.CAPABILITY_ARCH.values())):
        if world_size not in loader.supported_world_sizes(arch):
            continue
        for rank in range(world_size):
            _policy, plan = plans.plan_for(geometry, layout, rank, arch=arch)
            counts = _destination_writes(plan)
            per_dest = plan.M * 3 * geometry.h_local * HEAD_DIM
            for dest in range(world_size):
                written = counts[dest]
                assert int((written == 1).sum()) == per_dest
                assert int(written.max()) == 1
            tickets: dict[tuple[int, int], int] = {}
            last: dict[tuple[int, int], int] = {}
            for row in plan.tiles.tolist():
                dest, word, ticket = row[2], row[6], row[7]
                if dest < 0:
                    continue
                tickets[(dest, word)] = tickets.get((dest, word), 0) + 1
                last[(dest, word)] = ticket
            assert all(tickets[key] == last[key] + 1 for key in tickets)
            assert (
                plan.num_words * world_size <= plan.words_capacity_per_dest * world_size
            )
            if plan.pair:
                assert plan.num_tiles % 2 == 0
                for first, second in zip(
                    plan.tiles[0::2].tolist(), plan.tiles[1::2].tolist(), strict=True
                ):
                    assert first[1] == second[1]


def _destination_writes_xband(plan: plans.TilePlan) -> dict[int, torch.Tensor]:
    """:func:`_destination_writes` of a cross-destination-band plan: each row writes its two heads, each to its own
    destination and slot base."""

    indices: dict[int, list[torch.Tensor]] = {dest: [] for dest in range(plan.world)}
    columns = torch.arange(HEAD_DIM)
    for row in plan.tiles.tolist():
        mt, _n0, dest, _kind, base_elem, row_stride, _word, _ticket, dest1, base1 = row
        m = torch.arange(mt * plans.BLOCK_M, min(plan.M, (mt + 1) * plans.BLOCK_M))
        b, s = m // plan.s_local, m % plan.s_local
        for head_dest, head_base in ((dest, base_elem), (dest1, base1)):
            if head_dest < 0:
                continue
            row_offset = head_base + b * plan.batch_stride + s * row_stride
            indices[head_dest].append(
                (row_offset[:, None] + columns[None, :]).flatten()
            )
    return {
        dest: torch.bincount(
            torch.cat(parts) if parts else torch.zeros(0, dtype=torch.int64),
            minlength=plan.slot_elems,
        )
        for dest, parts in indices.items()
    }


@pytest.mark.parametrize("layout", plans.LAYOUTS)
@pytest.mark.parametrize("order", ["nt_outer", "grouped"])
@pytest.mark.parametrize(
    "world_size, batch, s_local, hidden",
    [(8, 1, 130, 1024), (8, 2, 64, 3072), (8, 1, 40, 5120)],
)
def test_xband_plan_writes_every_destination_element_once(
    layout, order, world_size, batch, s_local, hidden
):
    """The cross-destination-band table (256-column pair tiles across band edges, ``chunk_plan(xband=True)``): every
    destination element of the rank's slab is written exactly once, each ``(destination, word)`` collects one ticket
    per head up to its ``last_ticket``, lockstep pairs share their columns, and the table keeps the band plan's
    readiness words and per-destination ticket count."""

    geometry = plans.Geometry(
        world_size=world_size, batch=batch, s_local=s_local, hidden=hidden
    )
    assert plans.xband_eligible(geometry) and plans.block_n_for(geometry) == 128
    m_tiles = plans.ceil_div(geometry.rows, plans.BLOCK_M)
    group_m = None if order == "nt_outer" else 2
    for rank in range(world_size):
        band = plans.chunk_plan(
            geometry, layout, rank, block_n=128, order=order, group_m=group_m, pair=True
        )
        plan = plans.chunk_plan(
            geometry,
            layout,
            rank,
            block_n=256,
            order=order,
            group_m=group_m,
            pair=True,
            xband=True,
        )
        assert plan.xband and plan.tile_fields == plans.TILE_FIELDS_XB
        assert tuple(plan.tiles.shape) == (plan.num_tiles, len(plans.TILE_FIELDS_XB))
        assert plan.num_tiles - plan.num_pad_rows == m_tiles * 3 * hidden // 256
        assert (plan.order, plan.group_m) == (band.order, band.group_m)
        assert (plan.num_words, plan.n_sub) == (band.num_words, band.n_sub)
        assert plan.summary()["xband"] is True and "xband" not in band.summary()
        counts = _destination_writes_xband(plan)
        per_dest = plan.M * 3 * geometry.h_local * HEAD_DIM
        for dest in range(world_size):
            assert int((counts[dest] == 1).sum()) == per_dest
            assert int(counts[dest].max()) == 1
        tickets: dict[tuple[int, int], int] = {}
        last: dict[tuple[int, int], int] = {}
        for row in plan.tiles.tolist():
            word, ticket = row[plans.TILE_FIELDS.index("word")], row[7]
            for head_dest in (row[2], row[plans.TILE_DEST1]):
                if head_dest < 0:
                    continue
                tickets[(head_dest, word)] = tickets.get((head_dest, word), 0) + 1
                last[(head_dest, word)] = ticket
        assert all(tickets[key] == last[key] + 1 for key in tickets)
        assert plan.num_tiles % 2 == 0
        for first, second in zip(
            plan.tiles[0::2].tolist(), plan.tiles[1::2].tolist(), strict=True
        ):
            assert first[1] == second[1]
        with pytest.raises(ValueError, match="persistent launch"):
            plans.dest_parts(plan)
    for bad in (
        dict(block_n=128, pair=True),
        dict(block_n=256, pair=False),
    ):
        with pytest.raises(ValueError, match="pair plan"):
            plans.chunk_plan(geometry, "bshd", 0, xband=True, **bad)
    wide = plans.Geometry(world_size=4, batch=1, s_local=64, hidden=4096)
    with pytest.raises(ValueError, match="destination band"):
        plans.chunk_plan(wide, "bshd", 0, block_n=256, pair=True, xband=True)


def test_xband_plan_for_keeps_the_band_policy():
    """``plan_for(xband=True)`` on an eligible pair row: the band policy's tile order and ``group_m`` with the
    256-column pair tiles; a row whose policy is not the pair tile is refused."""

    for arch in ("sm_100a", "sm_103a"):
        for hidden, s_local in ((3072, 4096), (3072, 16384), (1024, 8192)):
            geometry = plans.Geometry(
                world_size=8, batch=1, s_local=s_local, hidden=hidden
            )
            policy, band = plans.plan_for(geometry, "bshd", 3, arch=arch)
            xb_policy, plan = plans.plan_for(geometry, "bshd", 3, arch=arch, xband=True)
            assert (policy.bn, policy.pair, band.bn) == (128, True, 128)
            assert (plan.bn, plan.pair, plan.xband) == (256, True, True)
            assert (plan.order, plan.group_m) == (policy.order, policy.group_m)
            assert (plan.num_words, plan.n_sub) == (band.num_words, band.n_sub)
            assert xb_policy.reasons[:-1] == policy.reasons
            assert "xband_min_m" in xb_policy.reasons[-1]
        small = plans.Geometry(world_size=8, batch=1, s_local=1024, hidden=3072)
        with pytest.raises(ValueError, match="pair tile"):
            plans.plan_for(small, "bshd", 0, arch=arch, xband=True)


def test_xband_below_pair_min_m_runs_the_grouped_pair_tile():
    """From ``XBAND_MIN_M_FLOOR`` to ``PAIR_MIN_M`` rows a fired cross-destination-band gate takes the pair geometry
    the band rule does not give: the grouped order with ``m_tiles // PAIR_MIN_GROUPS`` token tiles per group; the band
    plan of the same row stays the single-CTA 128-column tile."""

    for arch in ("sm_100a", "sm_103a"):
        geometry = plans.Geometry(world_size=8, batch=1, s_local=2048, hidden=3072)
        band_policy = plans.plan_policy(geometry, arch=arch)
        assert (band_policy.bn, band_policy.pair) == (128, False)
        xb_policy = plans.plan_policy(geometry, arch=arch, xband=True)
        assert (xb_policy.bn, xb_policy.pair, xb_policy.order, xb_policy.group_m) == (
            128,
            True,
            "grouped",
            4,
        )
        gates = plans.default_shape_gates(arch, world_size=8)
        verdict, policy, plan = plans.plan_row(
            geometry,
            "bshd",
            1,
            arch=arch,
            gates=gates,
            k_table=plans.default_shape_gate_k_table(arch, world_size=8),
            xband_min_m=plans.default_xband_min_m(arch, world_size=8),
        )
        assert (
            verdict.xband is True
            and plan.xband
            and (plan.bn, plan.pair, plan.group_m) == (256, True, 4)
        )
        below = plans.Geometry(world_size=8, batch=1, s_local=2047, hidden=3072)
        assert not plans.plan_policy(below, arch=arch, xband=True).pair


def test_blackwell_bn128_pair_tile_table():
    """``BLACKWELL_BN128_MAX_M_BY_ARCH_WORLD`` (SM100 / SM103 with 8 ranks at hidden 4096, up to 512 rows): a pair row
    of the cell runs the 128-column pair tile with the band policy's order and ``group_m``; a larger row, another
    cell and ``bn128_max_m=0`` keep the 256-column pair tile."""

    assert plans.BLACKWELL_BN128_MAX_M_BY_ARCH_WORLD == {
        ("sm_100a", 8): {4096: 512},
        ("sm_103a", 8): {4096: 512},
    }
    for arch in ("sm_100a", "sm_103a"):
        assert plans.default_bn128_max_m(arch, 8, 4096) == 512
        assert plans.default_bn128_max_m(arch, 4, 4096) == 0
        geometry = plans.Geometry(world_size=8, batch=1, s_local=512, hidden=4096)
        band = plans.plan_policy(geometry, arch=arch)
        assert (band.bn, band.pair) == (256, True)
        policy, plan = plans.plan_for(geometry, "bshd", 2, arch=arch)
        assert (policy.bn, policy.pair, plan.bn, plan.pair) == (128, True, 128, True)
        assert (plan.order, plan.group_m) == (band.order, band.group_m)
        off_policy, off = plans.plan_for(geometry, "bshd", 2, arch=arch, bn128_max_m=0)
        assert (off.bn, off.pair) == (256, True)
        _p, single = plans.plan_for(
            geometry, "bshd", 2, arch=arch, pair_available=False
        )
        assert (single.bn, single.pair) == (256, False)
        larger = plans.Geometry(world_size=8, batch=1, s_local=1024, hidden=4096)
        assert plans.plan_for(larger, "bshd", 2, arch=arch)[1].bn == 256
    assert plans.default_bn128_max_m("sm_90a", 4, 3072) == 1024


def test_dest_interleave_orders_the_table():
    """The destination interleave (``DEST_INTERLEAVE_MIN_M_BY_ARCH_WORLD`` bands, ``DEST_INTERLEAVE_DEFAULT_BY_ARCH_WORLD``
    modes): every mode keeps the set of tile rows (each row's destination, word and ticket) and only reorders them;
    ``plan_row`` applies the default mode on a band row, where a fired per-destination launch gate yields to it, and
    keeps the destination blocks off the band and above the cap."""

    assert plans.DEST_INTERLEAVE_MODES == ("off", "il", "lt", "ls")
    assert plans.resolve_dest_interleave("sm_100a", 2) == ("lt", True, 16384)
    assert plans.resolve_dest_interleave("sm_103a", 2) == ("lt", True, 16384)
    assert plans.resolve_dest_interleave("sm_103a", 4) == ("lt", True, None)
    assert plans.resolve_dest_interleave("sm_90a", 4) == ("il", False, None)
    assert plans.resolve_dest_interleave("sm_90a", 2) == ("il", False, None)
    assert sorted(plans.DEST_INTERLEAVE_DEFAULT_WITHOUT_PD) == [
        ("sm_90a", 2),
        ("sm_90a", 4),
    ]
    assert plans.resolve_dest_interleave("sm_107a", 4) == ("off", False, None)
    assert plans.resolve_dest_interleave("sm_90a", 4, "il") == ("il", False, None)
    with pytest.raises(ValueError, match="refused on sm_107a"):
        plans.resolve_dest_interleave("sm_107a", 4, "lt")
    with pytest.raises(ValueError, match="dest_interleave_pd"):
        plans.resolve_dest_interleave("sm_100a", 4, "off", pd=True)
    geometry = plans.Geometry(world_size=4, batch=1, s_local=32768, hidden=3072)
    _policy, base = plans.plan_for(geometry, "bshd", 1, arch="sm_103a")
    key = lambda table: sorted(map(tuple, table.tolist()))  # noqa: E731
    for mode in ("il", "lt", "ls"):
        _policy, plan = plans.plan_for(
            geometry, "bshd", 1, arch="sm_103a", dest_interleave=mode
        )
        assert plan.dest_interleave == mode and plan.num_tiles == base.num_tiles
        assert key(plan.tiles) == key(base.tiles) and not torch.equal(
            plan.tiles, base.tiles
        )
        assert (plan.interleave_tail_rows > 0) is (mode == "lt")
        with pytest.raises(ValueError, match="per-destination runs"):
            plans.dest_parts(plan)
    _policy, lt = plans.plan_for(
        geometry, "bshd", 1, arch="sm_103a", dest_interleave="lt"
    )
    assert lt.interleave_tail_rows == 146 - 146 % lt.lockstep
    assert int(lt.tiles[-1, plans.TILE_DEST]) in (1, -1)
    gates = plans.default_shape_gates("sm_103a", world_size=4)
    k_table = plans.default_shape_gate_k_table("sm_103a", world_size=4)
    verdict, _policy, row = plans.plan_row(
        geometry, "bshd", 1, arch="sm_103a", gates=gates, k_table=k_table
    )
    assert row.dest_interleave == "lt" and verdict.producer_launches == "persistent"
    assert verdict.dest_interleave_suppressed == ("per_dest_large_m",)
    assert verdict.summary()["dest_interleave_suppressed"] == ["per_dest_large_m"]
    off_verdict, _policy, off = plans.plan_row(
        geometry,
        "bshd",
        1,
        arch="sm_103a",
        gates=gates,
        k_table=k_table,
        dest_interleave="off",
    )
    assert off.dest_interleave == "off" and off_verdict.producer_launches == "per_dest"
    no_pd, _policy, kept = plans.plan_row(
        geometry,
        "bshd",
        1,
        arch="sm_103a",
        gates=gates,
        k_table=k_table,
        dest_interleave="lt",
    )
    assert no_pd.producer_launches == "per_dest" and kept.dest_interleave == "off"
    capped = plans.Geometry(world_size=2, batch=1, s_local=32768, hidden=3072)
    _v, _p, above = plans.plan_row(
        capped,
        "bshd",
        0,
        arch="sm_100a",
        gates=plans.default_shape_gates("sm_100a", world_size=2),
        k_table=plans.default_shape_gate_k_table("sm_100a", world_size=2),
    )
    assert above.dest_interleave == "off"
    small = plans.Geometry(world_size=4, batch=1, s_local=4096, hidden=3072)
    assert (
        plans.plan_row(small, "bshd", 0, arch="sm_103a", gates=gates, k_table=k_table)[
            2
        ].dest_interleave
        == "off"
    )
    # SM90 group of 4 (HP1): the band rows take ``il`` alone (no per-destination
    # launch precedence, no local tail); the rows below the band keep the blocks.
    h_gates = plans.default_shape_gates("sm_90a", world_size=4)
    h_k_table = plans.default_shape_gate_k_table("sm_90a", world_size=4)
    for s_local, mode in ((8192, "il"), (2048, "off")):
        hopper = plans.Geometry(world_size=4, batch=1, s_local=s_local, hidden=3072)
        h_verdict, _policy, h_row = plans.plan_row(
            hopper, "bshd", 1, arch="sm_90a", gates=h_gates, k_table=h_k_table
        )
        assert h_row.dest_interleave == mode and h_row.interleave_tail_rows == 0
        assert h_verdict.dest_interleave_suppressed == ()


def test_hopper_bn128_tile_table():
    """The 128-column tile table (``HOPPER_BN128_MAX_M_BY_ARCH_WORLD``: SM90 with 4 ranks at hidden 3072, up to 1024
    rows): a row of the cell at or below the bound runs the 128-column tile with the band policy's tile order,
    ``group_m`` and pair flag; a larger row, another cell and ``bn128_max_m=0`` keep the band tile; ``plan_row``
    reads its gates with the band width."""

    assert plans.HOPPER_BN128_MAX_M_BY_ARCH_WORLD == {("sm_90a", 4): {3072: 1024}}
    assert plans.HOPPER_BN128_BN == 128
    assert plans.hopper_bn128_max_m("sm_90a", 4, 3072) == 1024
    for arch, world_size, hidden in (
        ("sm_90a", 4, 4096),
        ("sm_90a", 4, 6144),
        ("sm_90a", 8, 3072),
        ("sm_90a", 2, 3072),
        ("sm_100a", 4, 3072),
        ("sm_103a", 4, 3072),
        ("sm_107a", 4, 3072),
    ):
        assert plans.hopper_bn128_max_m(arch, world_size, hidden) == 0
    gates = plans.default_shape_gates("sm_90a", world_size=4)
    for s_local, swapped in (
        (1, True),
        (256, True),
        (1024, True),
        (1025, False),
        (2048, False),
    ):
        geometry = plans.Geometry(world_size=4, batch=1, s_local=s_local, hidden=3072)
        assert plans.block_n_for(geometry) == 256
        band_policy, band = plans.plan_for(
            geometry,
            "bshd",
            1,
            arch="sm_90a",
            bn128_max_m=0,
        )
        policy, plan = plans.plan_for(geometry, "bshd", 1, arch="sm_90a")
        assert band_policy.bn == band.bn == 256
        assert policy.bn == plan.bn == (128 if swapped else 256)
        assert (policy.order, policy.group_m, policy.pair) == (
            band_policy.order,
            band_policy.group_m,
            band_policy.pair,
        )
        assert (plan.order, plan.group_m, plan.pair) == (
            band.order,
            band.group_m,
            band.pair,
        )
        if swapped:
            assert policy.reasons[:-1] == band_policy.reasons
            assert "HOPPER_BN128_MAX_M_BY_ARCH_WORLD" in policy.reasons[-1]
            assert plan.num_tiles == 2 * band.num_tiles
        else:
            assert policy == band_policy
        verdict, row_policy, row_plan = plans.plan_row(
            geometry, "bshd", 1, arch="sm_90a", gates=gates
        )
        assert verdict.bn == 256 and row_policy == policy and row_plan.bn == plan.bn
        explicit_policy, _plan = plans.plan_for(
            geometry, "bshd", 1, arch="sm_90a", bn128_max_m=1024
        )
        assert explicit_policy == policy
    # The swapped plan writes every destination element once, like the band plan.
    geometry = plans.Geometry(world_size=4, batch=1, s_local=200, hidden=3072)
    for rank in range(4):
        _policy, plan = plans.plan_for(geometry, "fused", rank, arch="sm_90a")
        assert plan.bn == 128
        counts = _destination_writes(plan)
        per_dest = plan.M * 3 * geometry.h_local * HEAD_DIM
        for dest in range(4):
            assert int((counts[dest] == 1).sum()) == per_dest
            assert int(counts[dest].max()) == 1
    for bad in (-1, True, 1.5, plans.SHAPE_GATE_ALL_ROWS):
        with pytest.raises(ValueError, match="bn128_max_m"):
            plans.plan_for(geometry, "bshd", 0, arch="sm_90a", bn128_max_m=bad)


def test_plan_policy_rules():
    small = plans.Geometry(world_size=8, batch=1, s_local=512, hidden=3072)
    assert plans.block_n_for(small) == 128
    policy = plans.plan_policy(small, arch="sm_100a")
    assert (policy.bn, policy.pair, policy.order) == (128, False, "nt_outer")
    wide = plans.Geometry(world_size=8, batch=1, s_local=512, hidden=4096)
    assert plans.block_n_for(wide) == 256
    assert plans.plan_policy(wide, arch="sm_100a").pair is True
    assert plans.plan_policy(wide, arch="sm_90a").pair is False
    assert plans.plan_policy(wide, arch="sm_100a", pair_available=False).pair is False
    assert plans.plan_policy(wide, arch="sm_100a").group_m_scale == 1
    large = plans.Geometry(world_size=2, batch=1, s_local=65536, hidden=8192)
    policy = plans.plan_policy(large, arch="sm_103a")
    assert policy.pair and policy.order == "grouped" and policy.group_m % 2 == 0
    # The (architecture, world size) default scale of a 2-rank group on SM103 doubles the token-tile group of this
    # grouped row (hidden 8192 >= GROUP_M_SCALE_MIN_K) over the explicit baseline scale.
    baseline = plans.plan_policy(large, arch="sm_103a", group_m_scale=1)
    assert policy.group_m_scale == plans.default_group_m_scale("sm_103a", 2) == 2
    assert baseline.group_m <= baseline.m_tiles // plans.PAIR_MIN_GROUPS
    doubled = min(policy.m_tiles, 2 * baseline.group_m)
    assert policy.group_m == doubled > baseline.group_m
    hopper = plans.plan_policy(large, arch="sm_90a")
    assert hopper.order == "grouped" and not hopper.pair
    assert hopper.group_m_scale == 1
    assert hopper == plans.plan_policy(large, arch="sm_90a", group_m_scale=1)
    # SM107 follows the tcgen05 policy (pair tile) on the world sizes it supports.
    rubin = plans.plan_policy(
        plans.Geometry(world_size=4, batch=1, s_local=1024, hidden=4096), arch="sm_107a"
    )
    assert rubin.pair is True and rubin.bn == 256
    assert "sm_107a" in plans.PAIR_ARCHES and "sm_90a" not in plans.PAIR_ARCHES


def test_geometry_rejects_heads_not_divisible_by_world_size():
    with pytest.raises(ValueError, match="divisible by world_size"):
        plans.Geometry(world_size=8, batch=1, s_local=128, hidden=3584)
    with pytest.raises(ValueError, match="divisible by world_size"):
        validate_request(
            world_size=8,
            hidden=3584,
            num_heads=28,
            head_dim=128,
            dtype=torch.bfloat16,
            layout="bshd",
            max_tokens=128,
            num_slots=2,
            backend="auto",
        )


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"world_size": 6}, "world size"),
        ({"dtype": torch.float16}, "bfloat16"),
        ({"head_dim": 64, "hidden": 512}, "head_dim"),
        ({"layout": "sbhd"}, "layout"),
        ({"backend": "ipc"}, "backend"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"num_slots": 1}, "num_slots"),
        ({"hidden": 2048}, "num_heads"),
        ({"fold_barrier": 1}, "fold_barrier must be a bool"),
        ({"fold_barrier": True, "backend": "nccl"}, "no meaning with backend='nccl'"),
    ],
)
def test_validate_request_rejects_unsupported_requests(overrides, match):
    request = dict(
        world_size=8,
        hidden=1024,
        num_heads=8,
        head_dim=128,
        dtype=torch.bfloat16,
        layout="bshd",
        max_tokens=256,
        num_slots=2,
        backend="auto",
    )
    request.update(overrides)
    with pytest.raises(ValueError, match=match):
        validate_request(**request)


def test_barrier_flag_words():
    for world_size in loader.SUPPORTED_WORLD_SIZES:
        assert loader.barrier_flag_words(world_size) == 2 + 4 * world_size


def test_supported_world_sizes_per_architecture():
    assert (
        set(loader.SUPPORTED_WORLD_SIZES_BY_ARCH)
        == set(loader.ARCH_FLAGS)
        == set(loader.CAPABILITY_ARCH.values())
    )
    for arch, sizes in loader.SUPPORTED_WORLD_SIZES_BY_ARCH.items():
        assert loader.supported_world_sizes(arch) == sizes
        assert set(sizes) <= set(loader.SUPPORTED_WORLD_SIZES)
    assert loader.supported_world_sizes("sm_107a") == (2, 4)
    assert loader.supported_world_sizes("sm_100a") == (2, 4, 8)
    with pytest.raises(ValueError, match="no fused QKV exchange programs"):
        loader.supported_world_sizes("sm_120a")


def test_fold_barrier_default_table():
    """The default launch path per architecture: the folded rendezvous on SM90, SM100, SM103
    and SM107 (Rubin, after its own measurement); the separate barrier kernel as
    the explicit ``fold_barrier=False`` path everywhere."""

    assert loader.FOLD_BARRIER_DEFAULT_BY_ARCH == {
        "sm_90a": True,
        "sm_100a": True,
        "sm_103a": True,
        "sm_107a": True,
    }
    assert set(loader.FOLD_BARRIER_DEFAULT_BY_ARCH) == set(loader.ARCH_FLAGS)
    for arch, fold in loader.FOLD_BARRIER_DEFAULT_BY_ARCH.items():
        assert loader.default_fold_barrier(arch) is fold
    with pytest.raises(ValueError, match="no default launch path"):
        loader.default_fold_barrier("sm_120a")
    assert loader.fold_producer_key(128, True, 1) == "producer_bn128_pair_fold1"
    assert loader.fold_producer_key(256, False, 0) == "producer_bn256_fold0"
    with pytest.raises(ValueError, match="rendezvous phase"):
        loader.fold_producer_key(128, False, 2)


def test_fold_barrier_resolution(monkeypatch):
    """Keyword over environment over architecture default; every delivered architecture folds
    by default and delivers the folded producers (SM107 included); ``True`` fails closed on a
    route table without them (modelled below by stripping one table)."""

    if not loader.ROUTES:
        pytest.skip("program tables are filled by the exporter")
    env = backend.FOLD_BARRIER_ENV
    for arch in loader.ROUTES:
        default = loader.default_fold_barrier(arch)
        assert loader.fold_producers_delivered(arch) is default
        monkeypatch.delenv(env, raising=False)
        assert backend.resolve_fold_barrier(arch) == (default, "default")
        assert backend.resolve_fold_barrier(arch, False) == (False, "keyword")
        monkeypatch.setenv(env, "auto")
        assert backend.resolve_fold_barrier(arch) == (default, "default")
        monkeypatch.setenv(env, "0")
        assert backend.resolve_fold_barrier(arch) == (False, "env")
        monkeypatch.setenv(env, "1")
        if default:
            assert backend.resolve_fold_barrier(arch) == (True, "env")
        else:
            with pytest.raises(
                ValueError, match="no producer program with the folded rendezvous"
            ):
                backend.resolve_fold_barrier(arch)
        monkeypatch.setenv(env, "0")
        if default:
            assert backend.resolve_fold_barrier(arch, True) == (True, "keyword")
        else:
            with pytest.raises(
                ValueError,
                match="pass fold_barrier=False to select the separate barrier kernel",
            ):
                backend.resolve_fold_barrier(arch, True)
        monkeypatch.setenv(env, "yes")
        with pytest.raises(ValueError, match=env):
            backend.resolve_fold_barrier(arch)
        monkeypatch.delenv(env, raising=False)
        with pytest.raises(ValueError, match="bool or None"):
            backend.resolve_fold_barrier(arch, 1)  # type: ignore[arg-type]
    # A route table without folded producers (a patched table of a fold-default architecture):
    # the default and the explicit ``True`` fail closed naming the opt-out, ``False`` resolves.
    fold_arches = [arch for arch in loader.ROUTES if loader.default_fold_barrier(arch)]
    if not fold_arches:
        pytest.skip("no fold-default architecture in the delivered route tables")
    arch = fold_arches[0]
    stripped = {
        key: value for key, value in loader.ROUTES[arch].items() if "_fold" not in key
    }
    monkeypatch.setattr(loader, "ROUTES", {**loader.ROUTES, arch: stripped})
    assert not loader.fold_producers_delivered(arch)
    with pytest.raises(
        ValueError, match="no producer program with the folded rendezvous"
    ):
        backend.resolve_fold_barrier(arch)
    with pytest.raises(
        ValueError,
        match="pass fold_barrier=False to select the separate barrier kernel",
    ):
        backend.resolve_fold_barrier(arch, True)
    assert backend.resolve_fold_barrier(arch, False) == (False, "keyword")


def test_shape_gate_default_table():
    """The row-count gates per architecture and world size, read from the tables themselves (no verbatim copy): the
    per-architecture table -- one readiness word per destination for calls of at most 4096 rows and one producer
    launch per destination run for calls of at least 16384 rows on SM100 / SM103; on SM90 one readiness word per
    destination for calls of at most 4096 rows, the persistent producer grid over every SM (no reserved SMs) for
    calls of at least 16384 rows, and the per-destination launch gate off; on SM107 no per-architecture gate -- plus
    the three (architecture, world size) cells that differ from it: the small-row word gate on every row for 8 ranks
    on SM90 and for 2 and 4 ranks on SM107."""

    assert plans.SHAPE_GATE_NAMES == (
        "words_capacity_small_m",
        "reserved_sms_large_m",
        "per_dest_large_m",
        "pair_off_large_m",
    )
    assert (
        plans.SHAPE_GATE_BN
        == {"pair_off_large_m": plans.PAIR_BN}
        == {"pair_off_large_m": 256}
    )
    assert plans.SHAPE_GATE_OFF == 0 and plans.RESERVED_SMS == 2
    assert plans.SHAPE_GATE_ALL_ROWS == 2**31 - 1 > 10**9
    assert plans.SHAPE_GATE_ALL_ROWS_NAMES == ("words_capacity_small_m",)
    table = plans.SHAPE_GATE_DEFAULT_BY_ARCH
    assert set(table) == set(loader.ARCH_FLAGS)
    for arch, row in table.items():
        assert tuple(row) == plans.SHAPE_GATE_NAMES
        assert plans.check_shape_gates(row) == row
        # The per-architecture semantics (the pair-tile gate is off in every per-architecture row).
        assert row["pair_off_large_m"] == plans.SHAPE_GATE_OFF
        if arch == "sm_107a":
            assert not any(row.values())
        else:
            assert row["words_capacity_small_m"] == 4096
            assert row["reserved_sms_large_m"] == (16384 if arch == "sm_90a" else 0)
            assert row["per_dest_large_m"] == (0 if arch == "sm_90a" else 16384)
    # The (architecture, world size) cells: exactly the new (architecture, world size) defaults, each differing from
    # its row.
    overrides = plans.SHAPE_GATE_DEFAULT_BY_ARCH_WORLD
    assert set(overrides) == {("sm_90a", 8), ("sm_107a", 2), ("sm_107a", 4)}
    for cell in overrides.values():
        assert cell == {"words_capacity_small_m": plans.SHAPE_GATE_ALL_ROWS}
    for (arch, world_size), cell in overrides.items():
        assert world_size in plans.SUPPORTED_WORLD_SIZES and set(cell) <= set(
            plans.SHAPE_GATE_NAMES
        )
        for name, value in cell.items():
            assert value != table[arch][name], "a cell equal to its row is refused"
    # ``default_shape_gates`` merges the two and never hands out the row alone.
    for arch in table:
        for world_size in plans.SUPPORTED_WORLD_SIZES:
            merged = plans.default_shape_gates(arch, world_size=world_size)
            assert tuple(merged) == plans.SHAPE_GATE_NAMES
            assert merged == {**table[arch], **overrides.get((arch, world_size), {})}
            assert (
                merged is not table[arch] and plans.check_shape_gates(merged) == merged
            )
            if (arch, world_size) not in overrides:
                assert merged == table[arch]
    assert plans.default_shape_gates("sm_90a", world_size=8) == {
        **table["sm_90a"],
        "words_capacity_small_m": plans.SHAPE_GATE_ALL_ROWS,
    }
    assert (
        plans.spell_shape_gates(plans.default_shape_gates("sm_90a", world_size=8))[
            "words_capacity_small_m"
        ]
        == "all"
    )
    assert plans.spell_shape_gates(table["sm_90a"]) == table["sm_90a"]
    with pytest.raises(TypeError):
        plans.default_shape_gates("sm_90a")  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="no shape gates"):
        plans.default_shape_gates("sm_120a", world_size=8)
    with pytest.raises(ValueError, match="world_size=16"):
        plans.default_shape_gates("sm_100a", world_size=16)
    # ``all`` round-trips through the table check (the small-row word gate only).
    assert plans.check_shape_gates(
        {**table["sm_90a"], "words_capacity_small_m": "all"}
    ) == (plans.default_shape_gates("sm_90a", world_size=8))
    for bad in (
        {**table["sm_90a"], "reserved_sms_large_m": "all"},
        {**table["sm_90a"], "per_dest_large_m": plans.SHAPE_GATE_ALL_ROWS},
        {"words_capacity_small_m": 1},
        {"words_capacity_small_m": 0, "per_dest_large_m": 0},
        {
            "words_capacity_small_m": 0,
            "reserved_sms_large_m": 0,
            "per_dest_large_m": 0,
        },
        {
            "words_capacity_small_m": -1,
            "reserved_sms_large_m": 0,
            "per_dest_large_m": 0,
            "pair_off_large_m": 0,
        },
        {
            "words_capacity_small_m": True,
            "reserved_sms_large_m": 0,
            "per_dest_large_m": 0,
            "pair_off_large_m": 0,
        },
        {
            "words_capacity_small_m": 0,
            "reserved_sms_large_m": 2.0,
            "per_dest_large_m": 0,
            "pair_off_large_m": 0,
        },
        {
            "words_capacity_small_m": 0,
            "reserved_sms_large_m": 0,
            "per_dest_large_m": 0,
            "pair_off_large_m": -8192,
        },
        # Above ``SHAPE_GATE_ALL_ROWS`` no row count is reached: ``all`` (every row, the small-row gate) and ``0``
        # (never) are the only spellings, on the word gate and on a large-row gate alike.
        {**table["sm_90a"], "words_capacity_small_m": plans.SHAPE_GATE_ALL_ROWS + 1},
        {**table["sm_90a"], "reserved_sms_large_m": 2**31},
        {**table["sm_100a"], "pair_off_large_m": 3_000_000_000},
    ):
        with pytest.raises(ValueError):
            plans.check_shape_gates(bad)
    # Inclusive thresholds; a gate at 0 never fires.
    blackwell = plans.default_shape_gates("sm_100a", world_size=8)
    assert plans.gate_rows(blackwell, 4096).fired == ("words_capacity_small_m",)
    assert plans.gate_rows(blackwell, 4096).words_capacity_per_dest == 1
    assert (
        plans.gate_rows(blackwell, 4097).fired
        == ()
        == plans.gate_rows(blackwell, 16383).fired
    )
    assert plans.gate_rows(blackwell, 16384).fired == ("per_dest_large_m",)
    assert plans.gate_rows(blackwell, 16384).producer_launches == "per_dest"
    assert plans.gate_rows(blackwell, 16384).words_capacity_per_dest is None
    assert plans.gate_rows(blackwell, 16384).reserved_sms is None
    hopper = plans.default_shape_gates("sm_90a", world_size=4)
    # SM90 ships the small-row word gate too (one readiness word per destination).
    for rows in (128, 4096):
        verdict = plans.gate_rows(hopper, rows)
        assert verdict.fired == ("words_capacity_small_m",)
        assert verdict.producer_launches == "persistent"
        assert verdict.words_capacity_per_dest == 1 and verdict.reserved_sms is None
        assert verdict.summary()["rows"] == rows
    verdict = plans.gate_rows(hopper, 16383)
    assert verdict.fired == () and verdict.producer_launches == "persistent"
    assert verdict.words_capacity_per_dest is None and verdict.reserved_sms is None
    for rows in (16384, 65536):
        verdict = plans.gate_rows(hopper, rows)
        assert verdict.fired == ("reserved_sms_large_m",)
        assert verdict.reserved_sms == 0 and verdict.producer_launches == "persistent"
        assert verdict.words_capacity_per_dest is None
        assert verdict.summary()["reserved_sms"] == 0
        assert verdict.summary()["fired"] == ["reserved_sms_large_m"]
    # SM90 with 8 ranks: the small-row word gate covers every row count (one readiness word per destination on
    # every call), beside the large-row reserved-SM gate; the verdict spells the threshold ``all``.
    hopper8 = plans.default_shape_gates("sm_90a", world_size=8)
    for rows in (1, 128, 4096, 4097, 16383, 16384, 65536, 10**8):
        verdict = plans.gate_rows(hopper8, rows)
        assert verdict.words_capacity_per_dest == 1
        assert (
            "words_capacity_small_m" in verdict.fired
        ) and verdict.producer_launches == "persistent"
        assert (verdict.reserved_sms == 0) is (rows >= 16384)
        assert verdict.summary()["gates"]["words_capacity_small_m"] == "all"
    assert plans.gate_rows(hopper8, 16384).fired == (
        "words_capacity_small_m",
        "reserved_sms_large_m",
    )
    # ``fired`` lists the gates in table order whatever fired.
    every = {
        "words_capacity_small_m": 20000,
        "reserved_sms_large_m": 16384,
        "per_dest_large_m": 16384,
        "pair_off_large_m": 16384,
    }
    assert plans.gate_rows(every, 16384, bn=256).fired == plans.SHAPE_GATE_NAMES
    assert plans.gate_rows(every, 16384, bn=256).reserved_sms == 0
    assert plans.gate_rows(every, 16384, bn=256).pair is False
    with pytest.raises(ValueError, match="positive int"):
        plans.gate_rows(blackwell, 0)
    # The pair-tile gate: inclusive at its threshold, 256-column calls only, needs the
    # tile width while it is on; off, the tile width is optional and nothing fires.
    opted = {**blackwell, "pair_off_large_m": 8192}
    for rows, bn, fires in (
        (8191, 256, False),
        (8192, 256, True),
        (65536, 256, True),
        (8192, 128, False),
        (65536, 128, False),
    ):
        verdict = plans.gate_rows(opted, rows, bn=bn)
        assert (verdict.pair is False) is fires and verdict.bn == bn
        assert ("pair_off_large_m" in verdict.fired) is fires
        assert verdict.summary()["pair"] == verdict.pair
        assert verdict.summary()["bn"] == bn
    assert plans.gate_rows(opted, 16384, bn=256).fired == (
        "per_dest_large_m",
        "pair_off_large_m",
    )
    with pytest.raises(ValueError, match="pair_off_large_m=8192 fires on bn == 256"):
        plans.gate_rows(opted, 8192)
    with pytest.raises(ValueError, match="bn must be one of"):
        plans.gate_rows(opted, 8192, bn=64)
    assert plans.gate_rows(blackwell, 65536).pair is None
    assert plans.gate_rows(blackwell, 65536, bn=256).pair is None
    assert plans.gate_rows(blackwell, 65536, bn=256).summary()["fired"] == [
        "per_dest_large_m"
    ]


def test_pair_tile_gate_selects_the_single_cta_plan_on_wide_large_rows():
    """A fired pair-tile gate runs a 256-column row of at least the threshold on the single-CTA producer with the
    pair policy's tile order and ``group_m`` (the plan is the pair plan with ``pair`` off: no lockstep pads);
    128-column rows and rows below the threshold keep the pair tile.  The fourth-gate plan equals the plan of the
    explicit ``pair=False`` override, so the opt-in path and the override path are one path."""

    gates = {
        **plans.default_shape_gates("sm_100a", world_size=2),
        "pair_off_large_m": 8192,
    }
    # The SM100 2-rank default is the opt-in form (the pair-tile gate ships off on every cell).
    assert plans.default_shape_gates("sm_100a", world_size=2) == {
        **gates,
        "pair_off_large_m": 0,
    }
    # (world_size, hidden, s_local) -> the pair policy's (order, group_m) on SM100 and SM103.
    wide_rows = {
        (4, 6144, 8192): ("grouped", 20),
        (2, 8192, 65536): ("grouped", 16),
    }
    for arch in ("sm_100a", "sm_103a"):
        for (world_size, hidden, s_local), (order, group_m) in wide_rows.items():
            geometry = plans.Geometry(
                world_size=world_size, batch=1, s_local=s_local, hidden=hidden
            )
            assert plans.block_n_for(geometry) == 256 and geometry.rows >= 8192
            verdict = plans.gate_rows(gates, geometry.rows, bn=256)
            assert verdict.pair is False and "pair_off_large_m" in verdict.fired
            policy, plan = plans.plan_for(geometry, "bshd", 0, arch=arch)
            gated_policy, gated = plans.plan_for(
                geometry, "bshd", 0, arch=arch, pair=verdict.pair
            )
            assert (policy.pair, policy.order, policy.group_m) == (True, order, group_m)
            assert (gated_policy.pair, gated_policy.order, gated_policy.group_m) == (
                False,
                order,
                group_m,
            )
            assert (plan.pair, plan.lockstep) == (True, 2)
            assert (gated.pair, gated.lockstep, gated.num_pad_rows) == (False, 1, 0)
            assert (gated.order, gated.group_m, gated.bn) == (order, group_m, 256)
            assert gated.num_words == plan.num_words
            assert gated_policy.reasons[:-1] == policy.reasons
            assert "pair override True -> False" in gated_policy.reasons[-1]
            # ``pair=None`` keeps the policy; an override equal to the policy adds nothing.
            same_policy, same = plans.plan_for(
                geometry, "bshd", 0, arch=arch, pair=True
            )
            assert same_policy == policy and torch.equal(same.tiles, plan.tiles)
        # Controls: a 128-column row at any size and a 256-column row below the threshold.
        narrow = plans.Geometry(world_size=8, batch=1, s_local=16384, hidden=3072)
        assert plans.block_n_for(narrow) == 128
        assert plans.gate_rows(gates, narrow.rows, bn=128).pair is None
        small = plans.Geometry(world_size=4, batch=1, s_local=1024, hidden=4096)
        assert plans.block_n_for(small) == 256
        assert plans.gate_rows(gates, small.rows, bn=256).pair is None
        for geometry in (narrow, small):
            _policy, plan = plans.plan_for(geometry, "bshd", 0, arch=arch)
            assert plan.pair is True
    # SM90 builds no pair tile: the plan is the single-CTA plan with the gate off.
    hopper = plans.Geometry(world_size=8, batch=1, s_local=16384, hidden=8192)
    policy, plan = plans.plan_for(
        hopper, "bshd", 0, arch="sm_90a", pair_available=False
    )
    assert (policy.pair, plan.pair, plan.lockstep) == (False, False, 1)
    assert (
        plans.gate_rows(
            plans.default_shape_gates("sm_90a", world_size=8), hopper.rows
        ).pair
        is None
    )


def test_pair_tile_gate_reaches_the_single_cta_producer_and_names_itself(monkeypatch):
    """With the pair-tile gate on, a 256-column session must load the single-CTA producer as well as the pair
    producer (``reachable_pairs``), and a delivery without that producer refuses at construction naming the gate --
    not the producer-variant variable.  Off, or below ``max_tokens``, or on a 128-column session, nothing changes."""

    if not loader.ROUTES:
        pytest.skip("program tables are filled by the exporter")
    arches = [arch for arch in loader.ROUTES if arch in plans.PAIR_ARCHES]
    if not arches:
        pytest.skip("no pair architecture in the delivered route tables")
    for arch in arches:
        default = plans.default_shape_gates(arch, world_size=4)
        gated = {**default, "pair_off_large_m": 8192}
        assert backend.reachable_pairs(arch, 256, True, default, 65536) == (True,)
        assert backend.reachable_pairs(arch, 256, True, gated, 65536) == (False, True)
        assert backend.reachable_pairs(arch, 256, True, gated, 8192) == (False, True)
        assert backend.reachable_pairs(arch, 256, True, gated, 8191) == (True,)
        assert backend.reachable_pairs(arch, 256, False, gated, 65536) == (False,)
        for max_tokens in (512, 4096, 65536):
            assert backend.reachable_pairs(
                arch, 128, True, gated, max_tokens
            ) == backend.reachable_pairs(arch, 128, True, default, max_tokens)
        # A table without the single-CTA 256-column producer (the delivered Blackwell routes ship the pair
        # producer only): the refusal names the gate, the delivered builds and the opt-out.
        stripped = {
            key: value
            for key, value in loader.ROUTES[arch].items()
            if not (
                key.startswith("producer_bn")
                and loader.parse_producer_key(key).block_n == 256
                and not loader.parse_producer_key(key).pair
            )
        }
        with monkeypatch.context() as patched:
            patched.setattr(loader, "ROUTES", {**loader.ROUTES, arch: stripped})
            assert not loader.has_producer(arch, 256, False)
            message = backend.undelivered_producer_error(
                arch, 256, False, "", "default", gated
            )
            assert "pair_off_large_m=8192" in message and "single-CTA" in message
            assert "block_n=256" in message and arch in message
            assert backend.PRODUCER_VARIANT_ENV not in message
            assert backend.SHAPE_GATES_ENV in message
            plain = backend.undelivered_producer_error(
                arch, 256, False, "", "default", default
            )
            assert (
                "pair_off_large_m" not in plain
                and backend.PRODUCER_VARIANT_ENV in plain
            )
    # SM90 builds no pair tile: every table reaches the single-CTA producer only.
    assert backend.reachable_pairs(
        "sm_90a", 256, False, plans.default_shape_gates("sm_90a", world_size=8), 65536
    ) == (False,)


def test_shape_gates_resolution(monkeypatch):
    """Environment over architecture default: ``auto`` keeps the table, ``off`` switches every gate off, a partial
    ``gate=value`` list overrides only the named gates; malformed values are errors."""

    env = backend.SHAPE_GATES_ENV
    cells = [
        (arch, world_size)
        for arch in plans.SHAPE_GATE_DEFAULT_BY_ARCH
        for world_size in plans.SUPPORTED_WORLD_SIZES
    ]
    for arch, world_size in cells:
        monkeypatch.delenv(env, raising=False)
        assert backend.resolve_shape_gates(arch, world_size) == (
            plans.default_shape_gates(arch, world_size=world_size),
            "default",
        )
        monkeypatch.setenv(env, "auto")
        assert backend.resolve_shape_gates(arch, world_size) == (
            plans.default_shape_gates(arch, world_size=world_size),
            "default",
        )
        monkeypatch.setenv(env, "off")
        assert backend.resolve_shape_gates(arch, world_size) == (
            {
                "words_capacity_small_m": 0,
                "reserved_sms_large_m": 0,
                "per_dest_large_m": 0,
                "pair_off_large_m": 0,
            },
            "env",
        )
        monkeypatch.setenv(env, "per_dest_large_m=0")
        gates, source = backend.resolve_shape_gates(arch, world_size)
        defaults = plans.default_shape_gates(arch, world_size=world_size)
        assert source == "env" and gates["per_dest_large_m"] == 0
        assert gates["words_capacity_small_m"] == defaults["words_capacity_small_m"]
        assert gates["reserved_sms_large_m"] == defaults["reserved_sms_large_m"]
        monkeypatch.setenv(env, "words_capacity_small_m=2048; per_dest_large_m=auto")
        gates, source = backend.resolve_shape_gates(arch, world_size)
        assert source == "env" and gates == {
            "words_capacity_small_m": 2048,
            "reserved_sms_large_m": defaults["reserved_sms_large_m"],
            "per_dest_large_m": defaults["per_dest_large_m"],
            "pair_off_large_m": defaults["pair_off_large_m"],
        }
        # The small-row word gate on every row (``all``) in either direction: the SM90 8-rank and SM107 2- and
        # 4-rank default, an opt-in elsewhere; the previous SM90 8-rank plan is the explicit threshold 4096.
        monkeypatch.setenv(env, "words_capacity_small_m=all")
        gates, source = backend.resolve_shape_gates(arch, world_size)
        assert (
            source == "env"
            and gates["words_capacity_small_m"] == plans.SHAPE_GATE_ALL_ROWS
        )
        assert (
            plans.gate_rows(gates, 10**6, bn=plans.PAIR_BN).words_capacity_per_dest == 1
        )
        assert gates == {
            **defaults,
            "words_capacity_small_m": plans.SHAPE_GATE_ALL_ROWS,
        }
        if (arch, world_size) == ("sm_90a", 8):
            assert gates == defaults
            monkeypatch.setenv(env, "words_capacity_small_m=4096")
            gates, source = backend.resolve_shape_gates(arch, world_size)
            assert source == "env" and gates == plans.default_shape_gates(
                arch, world_size=4
            )
            assert plans.gate_rows(gates, 4097).words_capacity_per_dest is None
        # The pair-tile gate: an opt-in on the pair architectures, refused where no
        # pair tile is built (nothing to switch off); its opt-out resolves everywhere.
        monkeypatch.setenv(env, "pair_off_large_m=8192")
        if arch in plans.PAIR_ARCHES:
            gates, source = backend.resolve_shape_gates(arch, world_size)
            assert source == "env" and gates["pair_off_large_m"] == 8192
            assert gates["words_capacity_small_m"] == defaults["words_capacity_small_m"]
            assert plans.gate_rows(gates, 8192, bn=256).pair is False
            assert plans.gate_rows(gates, 8192, bn=128).pair is None
        else:
            with pytest.raises(ValueError, match="pair_off_large_m=8192 is refused on"):
                backend.resolve_shape_gates(arch, world_size)
        monkeypatch.setenv(env, "pair_off_large_m=0")
        gates, source = backend.resolve_shape_gates(arch, world_size)
        assert source == "env" and gates == {**defaults, "pair_off_large_m": 0}
        monkeypatch.setenv(env, "pair_off_large_m=auto")
        assert backend.resolve_shape_gates(arch, world_size) == (defaults, "env")
        # The large-row reserved-SM gate in either direction: armed at any row count, or switched off.
        monkeypatch.setenv(env, "reserved_sms_large_m=1")
        gates, source = backend.resolve_shape_gates(arch, world_size)
        assert source == "env" and gates["reserved_sms_large_m"] == 1
        assert plans.gate_rows(gates, 1, bn=plans.PAIR_BN).reserved_sms == 0
        monkeypatch.setenv(env, "reserved_sms_large_m=0")
        gates, source = backend.resolve_shape_gates(arch, world_size)
        assert source == "env" and gates["reserved_sms_large_m"] == 0
        assert plans.gate_rows(gates, 65536, bn=plans.PAIR_BN).reserved_sms is None
        for bad in (
            "yes",
            "words_capacity_small_m",
            "grid_small_m=1",
            "per_dest_large_m=-3",
            "pair_off_large_m=-1",
            "pair_off_large_m=x",
            "per_dest_large_m=x",
            "reserved_sms_large_m=on",
            "reserved_sms_large_m=all",
            "per_dest_large_m=all",
            "pair_off_large_m=all",
        ):
            monkeypatch.setenv(env, bad)
            with pytest.raises(ValueError, match=env):
                backend.resolve_shape_gates(arch, world_size)
    monkeypatch.delenv(env, raising=False)
    assert backend.shape_gates_from_env() is None


def test_shape_gate_k_table_and_xband_defaults():
    """The per-destination launch threshold by hidden size (``SHAPE_GATE_K_TABLE``; SM100 / SM103) and the
    cross-destination-band gate (``XBAND_MIN_M_DEFAULT_BY_ARCH_WORLD``; SM103 and SM100 with 8 ranks): the tables, their
    lookups and the verdicts ``gate_rows`` / ``plan_row`` derive from them."""

    assert plans.SHAPE_GATE_K_TABLE == {
        ("sm_100a", 8): ((0, 16384), (4096, 32768), (6144, 16384)),
        ("sm_100a", 4): ((0, 32768), (6144, 16384), (8192, 8192)),
        ("sm_100a", 2): ((0, 16384), (8192, 8192)),
        ("sm_103a", 8): ((0, 32768), (6144, 16384)),
        ("sm_103a", 4): ((0, 32768), (6144, 16384)),
        ("sm_103a", 2): ((0, 32768), (4096, 16384)),
    }
    assert plans.XBAND_MIN_M_DEFAULT_BY_ARCH_WORLD == {
        ("sm_103a", 8): 2048,
        ("sm_100a", 8): 2048,
    }
    assert plans.XBAND_MIN_M_FLOOR == 2048
    assert plans.XBAND_BN == plans.PAIR_BN == 256
    xb_fields = plans.TILE_FIELDS_XB
    assert xb_fields == (*plans.TILE_FIELDS, "dest1", "base_elem1")
    for arch in plans.SHAPE_GATE_DEFAULT_BY_ARCH:
        for world_size in plans.SUPPORTED_WORLD_SIZES:
            table = plans.default_shape_gate_k_table(arch, world_size=world_size)
            assert table == plans.SHAPE_GATE_K_TABLE.get((arch, world_size))
            if table is not None:
                assert plans.check_shape_gate_k_table(table) == table
            assert plans.default_xband_min_m(
                arch, world_size=world_size
            ) == plans.XBAND_MIN_M_DEFAULT_BY_ARCH_WORLD.get((arch, world_size), 0)
    assert plans.default_shape_gate_k_table("sm_90a", world_size=8) is None
    assert plans.default_shape_gate_k_table("sm_107a", world_size=4) is None
    with pytest.raises(ValueError, match="world_size=16"):
        plans.default_shape_gate_k_table("sm_100a", world_size=16)
    with pytest.raises(ValueError, match="no shape gates"):
        plans.default_xband_min_m("sm_120a", world_size=8)
    for bad in (
        (),
        ((1, 16384),),
        ((0, 16384), (0, 8192)),
        ((0, 16384), (4096, 16384)),
        ((0, -1),),
        ((0, plans.SHAPE_GATE_ALL_ROWS),),
        ((0, True),),
    ):
        with pytest.raises(ValueError):
            plans.check_shape_gate_k_table(bad)
    # The band of a hidden size is the last band whose lower bound it reaches.
    sm100_8 = plans.SHAPE_GATE_K_TABLE[("sm_100a", 8)]
    for hidden, threshold in (
        (1024, 16384),
        (3072, 16384),
        (4096, 32768),
        (5120, 32768),
        (6144, 16384),
        (8192, 16384),
    ):
        assert plans.k_band_threshold(sm100_8, hidden) == threshold
    gates = plans.default_shape_gates("sm_100a", world_size=8)
    unsplit = plans.gate_rows(gates, 16384, bn=256, hidden=4096, k_table=sm100_8)
    assert unsplit.producer_launches == "persistent" and unsplit.fired == ()
    assert unsplit.per_dest_threshold == 32768
    assert unsplit.summary()["per_dest_k_table"] == [list(band) for band in sm100_8]
    split = plans.gate_rows(gates, 32768, bn=256, hidden=4096, k_table=sm100_8)
    assert split.producer_launches == "per_dest" and split.fired == (
        "per_dest_large_m",
    )
    assert plans.gate_rows(
        gates, 16384, bn=256, hidden=8192, k_table=sm100_8
    ).fired == ("per_dest_large_m",)
    # Without a table the verdict and its record are the one-band ones.
    one_band = plans.gate_rows(gates, 16384, bn=256, hidden=4096)
    assert one_band.producer_launches == "per_dest"
    assert "per_dest_k_table" not in one_band.summary()
    with pytest.raises(ValueError, match="hidden"):
        plans.gate_rows(gates, 16384, bn=256, k_table=sm100_8)
    # The cross-destination-band gate: eligible geometries, thresholds, precedence over the per-destination launch.
    for world_size, hidden, eligible in (
        (8, 3072, True),
        (8, 1024, True),
        (8, 5120, True),
        (8, 4096, False),
        (4, 3072, False),
        (2, 8192, False),
    ):
        geometry = plans.Geometry(
            world_size=world_size, batch=1, s_local=4096, hidden=hidden
        )
        assert plans.xband_eligible(geometry) is eligible
    assert plans.check_xband_min_m("sm_103a", 0) == 0
    assert plans.check_xband_min_m("sm_100a", 8192) == 8192
    for arch, value in (
        ("sm_90a", 4096),
        ("sm_103a", 2047),
        ("sm_103a", -1),
        ("sm_103a", True),
        ("sm_103a", plans.SHAPE_GATE_ALL_ROWS),
    ):
        with pytest.raises(ValueError, match="xband_min_m"):
            plans.check_xband_min_m(arch, value)
    sm103_8 = plans.SHAPE_GATE_K_TABLE[("sm_103a", 8)]
    gates = plans.default_shape_gates("sm_103a", world_size=8)

    def verdict(rows, hidden=3072, **kwargs):
        return plans.gate_rows(
            gates, rows, bn=128, hidden=hidden, k_table=sm103_8, **kwargs
        )

    assert verdict(4095, xband_min_m=4096).xband is False
    with pytest.raises(ValueError, match="xband_eligible"):
        verdict(4096, xband_min_m=4096)
    fired = verdict(4096, xband_min_m=4096, xband_eligible=True)
    assert fired.xband is True and fired.fired == (
        "words_capacity_small_m",
        "xband_min_m",
    )
    assert fired.summary()["xband"] is True and fired.summary()["xband_min_m"] == 4096
    large = verdict(32768, xband_min_m=4096, xband_eligible=True)
    assert large.xband is True and large.producer_launches == "persistent"
    assert large.xband_suppressed == ("per_dest_large_m",)
    assert verdict(32768, xband_min_m=4096, xband_eligible=False).producer_launches == (
        "per_dest"
    )
    assert verdict(32768).xband is None and "xband" not in verdict(32768).summary()
    assert (
        plans.gate_rows(gates, 8192, bn=256, hidden=4096, xband_min_m=4096).xband
        is False
    )
    # ``plan_row``: the verdict, the policy and the plan of one call.
    xb_geometry = plans.Geometry(world_size=8, batch=1, s_local=8192, hidden=3072)
    row_verdict, policy, plan = plans.plan_row(
        xb_geometry,
        "bshd",
        2,
        arch="sm_103a",
        gates=gates,
        k_table=sm103_8,
        xband_min_m=4096,
        dest_interleave="off",
    )
    assert row_verdict.xband is True and plan.xband and plan.bn == 256
    assert policy.bn == 128 and policy.pair
    band_verdict, band_policy, band = plans.plan_row(
        xb_geometry,
        "bshd",
        2,
        arch="sm_103a",
        gates=gates,
        k_table=sm103_8,
        dest_interleave="off",
    )
    assert band_verdict.xband is None and not band.xband and band.bn == 128
    assert (plan.order, plan.group_m) == (band.order, band.group_m)
    assert band_policy.reasons == policy.reasons[:-1]


def test_shape_gate_k_table_and_xband_resolution(monkeypatch):
    """``per_dest_k_table`` and ``xband_min_m`` of ``FLASHINFER_QKV_EXCHANGE_SHAPE_GATES``: ``auto`` / unset keep
    the (architecture, world size) default, ``0`` switches the table / gate off, ``off`` switches both off, an
    explicit ``per_dest_large_m`` replaces the table; malformed or refused values are errors naming the variable."""

    env = backend.SHAPE_GATES_ENV
    for arch in plans.SHAPE_GATE_DEFAULT_BY_ARCH:
        for world_size in plans.SUPPORTED_WORLD_SIZES:
            table = plans.default_shape_gate_k_table(arch, world_size=world_size)
            xband = plans.default_xband_min_m(arch, world_size=world_size)
            monkeypatch.delenv(env, raising=False)
            assert backend.resolve_shape_gate_k_table(arch, world_size) == (
                table,
                "default",
            )
            assert backend.resolve_xband_min_m(arch, world_size) == (xband, "default")
            monkeypatch.setenv(env, "off")
            assert backend.resolve_shape_gate_k_table(arch, world_size) == (None, "env")
            assert backend.resolve_xband_min_m(arch, world_size) == (0, "env")
            monkeypatch.setenv(env, "per_dest_k_table=0,xband_min_m=0")
            assert backend.resolve_shape_gate_k_table(arch, world_size) == (None, "env")
            assert backend.resolve_xband_min_m(arch, world_size) == (0, "env")
            # The four gates keep their default when only these keys are named.
            assert backend.resolve_shape_gates(arch, world_size) == (
                plans.default_shape_gates(arch, world_size=world_size),
                "default",
            )
            monkeypatch.setenv(env, "per_dest_k_table=auto,xband_min_m=auto")
            assert backend.resolve_shape_gate_k_table(arch, world_size) == (
                table,
                "default",
            )
            assert backend.resolve_xband_min_m(arch, world_size) == (xband, "default")
            monkeypatch.setenv(env, "per_dest_large_m=8192")
            assert backend.resolve_shape_gate_k_table(arch, world_size) == (
                None,
                "per_dest_large_m",
            )
            monkeypatch.setenv(env, "per_dest_k_table=1")
            if table is None:
                with pytest.raises(ValueError, match="per_dest_k_table=1 is refused"):
                    backend.resolve_shape_gate_k_table(arch, world_size)
            else:
                assert backend.resolve_shape_gate_k_table(arch, world_size) == (
                    table,
                    "env",
                )
                monkeypatch.setenv(env, "per_dest_k_table=1,per_dest_large_m=8192")
                with pytest.raises(ValueError, match="cannot be combined"):
                    backend.resolve_shape_gate_k_table(arch, world_size)
            monkeypatch.setenv(env, "xband_min_m=8192")
            if arch in plans.PAIR_ARCHES:
                assert backend.resolve_xband_min_m(arch, world_size) == (8192, "env")
            else:
                with pytest.raises(ValueError, match=env):
                    backend.resolve_xband_min_m(arch, world_size)
            for bad in ("xband_min_m=2047", "xband_min_m=x", "xband_min_m=-1"):
                monkeypatch.setenv(env, bad)
                with pytest.raises(ValueError, match=env):
                    backend.resolve_xband_min_m(arch, world_size)
            for bad in ("per_dest_k_table=2", "per_dest_k_table=on"):
                monkeypatch.setenv(env, bad)
                with pytest.raises(ValueError, match=env):
                    backend.resolve_shape_gate_k_table(arch, world_size)
    monkeypatch.delenv(env, raising=False)
    assert backend.shape_gate_extras_from_env() == {
        "per_dest_k_table": None,
        "xband_min_m": None,
    }


def test_group_m_scale_default_table():
    """The group scaling per architecture and world size: the token-tile group of a grouped row with a hidden size of
    at least 6144 is doubled on SM100 groups of 4 and 2 ranks and on SM103, kept on SM100 groups of 8 ranks, on
    SM90 and on SM107; ``plan_policy`` without an explicit scale resolves the geometry's world size against the
    table."""

    assert plans.SUPPORTED_WORLD_SIZES == loader.SUPPORTED_WORLD_SIZES == (2, 4, 8)
    assert plans.GROUP_M_SCALES == (1, 2, 4) and plans.GROUP_M_SCALE_MIN_K == 6144
    assert plans.GROUP_M_SCALE_DEFAULT_BY_ARCH_WORLD == {
        "sm_90a": {2: 1, 4: 1, 8: 1},
        "sm_100a": {2: 2, 4: 2, 8: 1},
        "sm_103a": {2: 2, 4: 2, 8: 2},
        # the baseline plan on SM107 (the 2-rank doubling was not adopted)
        "sm_107a": {2: 1, 4: 1, 8: 1},
    }
    assert set(plans.GROUP_M_SCALE_DEFAULT_BY_ARCH_WORLD) == set(loader.ARCH_FLAGS)
    for arch, row in plans.GROUP_M_SCALE_DEFAULT_BY_ARCH_WORLD.items():
        assert tuple(row) == plans.SUPPORTED_WORLD_SIZES
        for world_size, scale in row.items():
            assert plans.default_group_m_scale(arch, world_size) == scale
            assert scale in plans.GROUP_M_SCALES
    with pytest.raises(ValueError, match="no group_m_scale default"):
        plans.default_group_m_scale("sm_120a", 8)
    with pytest.raises(ValueError, match="world_size=16"):
        plans.default_group_m_scale("sm_100a", 16)
    with pytest.raises(ValueError, match="world_size=1"):
        plans.default_group_m_scale("sm_90a", 1)
    # A grouped row with hidden 8192 at every world size: the default policy equals the explicit default scale,
    # and scales the baseline (scale 1) group within the tile count.
    for world_size in plans.SUPPORTED_WORLD_SIZES:
        geometry = plans.Geometry(
            world_size=world_size, batch=1, s_local=131072 // world_size, hidden=8192
        )
        for arch in plans.SHAPE_GATE_DEFAULT_BY_ARCH:
            pair_available = arch in plans.PAIR_ARCHES
            scale = plans.default_group_m_scale(arch, world_size)
            default = plans.plan_policy(
                geometry, arch=arch, pair_available=pair_available
            )
            explicit = plans.plan_policy(
                geometry, arch=arch, pair_available=pair_available, group_m_scale=scale
            )
            baseline = plans.plan_policy(
                geometry, arch=arch, pair_available=pair_available, group_m_scale=1
            )
            assert default == explicit and default.order == "grouped"
            assert default.group_m_scale == scale and baseline.group_m_scale == 1
            assert default.group_m == min(default.m_tiles, scale * baseline.group_m)
            assert (default.group_m != baseline.group_m) == (scale != 1)
            if default.pair:
                assert default.group_m % plans.PAIR_LOCKSTEP == 0
    # Below the hidden-size threshold the scale leaves the plan alone.
    small_k = plans.Geometry(world_size=2, batch=1, s_local=65536, hidden=4096)
    for arch in plans.PAIR_ARCHES:
        scaled = plans.plan_policy(small_k, arch=arch)
        baseline = plans.plan_policy(small_k, arch=arch, group_m_scale=1)
        # SM100 / SM103 ship scale 2 at W = 2; SM107 keeps the baseline scale 1 on every W.
        assert (
            scaled.group_m_scale
            == plans.default_group_m_scale(arch, 2)
            == (1 if arch == "sm_107a" else 2)
        )
        assert scaled.order == "grouped"
        assert scaled.group_m == baseline.group_m


def test_group_m_scale_resolution(monkeypatch):
    """Environment over the (architecture, world size) default: ``auto`` keeps the table; ``1`` / ``2`` / ``4``
    select a scale; anything else is an error naming the variable."""

    env = backend.GROUP_M_SCALE_ENV
    resolve = backend.resolve_group_m_scale
    for arch in plans.GROUP_M_SCALE_DEFAULT_BY_ARCH_WORLD:
        for world_size in plans.SUPPORTED_WORLD_SIZES:
            default = plans.default_group_m_scale(arch, world_size)
            monkeypatch.delenv(env, raising=False)
            assert resolve(arch, world_size) == (default, "default")
            monkeypatch.setenv(env, "auto")
            assert resolve(arch, world_size) == (default, "default")
            monkeypatch.setenv(env, "")
            assert resolve(arch, world_size) == (default, "default")
            for scale in plans.GROUP_M_SCALES:
                monkeypatch.setenv(env, str(scale))
                assert resolve(arch, world_size) == (scale, "env")
            for bad in ("3", "x", "0", "-1", "2.0", "shipped"):
                monkeypatch.setenv(env, bad)
                with pytest.raises(ValueError, match=env):
                    resolve(arch, world_size)
    monkeypatch.delenv(env, raising=False)
    assert backend.group_m_scale_from_env() is None
    with pytest.raises(ValueError, match="world_size=16"):
        resolve("sm_100a", 16)


def test_producer_key_grammar():
    """Producer route keys: ``producer_bn<bn>[_pair][_<variant>][_fold<phase>]``; ``parse_producer_key`` is the
    inverse and refuses every other spelling."""

    assert loader.PRODUCER_VARIANTS == (
        "",
        "ow256",
        "bulk_ow256",
        "bulk",
        "fma",
        "ow256_fma",
        "bulk_ow256_fma",
        "sw3",
    )
    assert loader.SHIPPED_VARIANT == ""
    assert loader.PRODUCER_VARIANT_OPERAND_LOAD_WIDTH == {
        "": 128,
        "ow256": 256,
        "bulk_ow256": 256,
        "bulk": 128,
        "fma": 128,
        "ow256_fma": 256,
        "bulk_ow256_fma": 256,
        "sw3": 128,
    }
    assert loader.producer_key(256, True) == "producer_bn256_pair"
    assert loader.producer_key(128, False) == "producer_bn128"
    assert (
        loader.producer_key(256, True, "bulk_ow256") == "producer_bn256_pair_bulk_ow256"
    )
    assert loader.producer_key(256, True, "bulk") == "producer_bn256_pair_bulk"
    assert loader.producer_key(128, False, "ow256") == "producer_bn128_ow256"
    assert loader.fold_producer_key(128, True, 1) == "producer_bn128_pair_fold1"
    assert loader.fold_producer_key(256, False, 0) == "producer_bn256_fold0"
    assert (
        loader.fold_producer_key(128, True, 1, "ow256")
        == "producer_bn128_pair_ow256_fold1"
    )
    assert (
        loader.fold_producer_key(256, True, 0, "bulk_ow256")
        == "producer_bn256_pair_bulk_ow256_fold0"
    )
    assert (
        loader.fold_producer_key(256, True, 1, "bulk")
        == "producer_bn256_pair_bulk_fold1"
    )
    assert (
        loader.producer_key(128, True, "bulk_ow256_fma")
        == "producer_bn128_pair_bulk_ow256_fma"
    )
    assert (
        loader.fold_producer_key(256, True, 0, "fma") == "producer_bn256_pair_fma_fold0"
    )
    parse = loader.parse_producer_key
    for block_n in plans.BLOCK_N_CHOICES:
        for pair in (False, True):
            for variant in loader.PRODUCER_VARIANTS:
                plain = loader.producer_key(block_n, pair, variant)
                assert parse(plain) == (block_n, pair, variant, None)
                for phase in loader.FOLD_PHASES:
                    fold = loader.fold_producer_key(block_n, pair, phase, variant)
                    parsed = parse(fold)
                    assert parsed == (block_n, pair, variant, phase)
                    assert parsed.block_n == block_n and parsed.fold_phase == phase
                    assert fold.rsplit("_fold", 1)[0] == plain
    for bad in (
        "wait",
        "producer",
        "producer_bn",
        "producer_bnx",
        "producer_bn128_pairx",
        "producer_bn128_ow512",
        "producer_bn128_fold2",
        "producer_bn128_pair_fold",
        "producer_bn128_bulk_ow256_pair",
        "producer_bn128_ow256_bulk",
    ):
        with pytest.raises(ValueError):
            parse(bad)
    with pytest.raises(ValueError, match="rendezvous phase"):
        loader.fold_producer_key(128, False, 2)
    with pytest.raises(ValueError, match="producer build variant"):
        loader.producer_key(128, False, "wide")
    # Cross-destination-band builds: ``<variant>_xb`` of the 256-column pair tile only, on the staged row stores (a
    # bulk build maps to the cross-destination-band build of its staged twin).
    assert loader.XBAND_SUFFIX == "xb" and loader.XBAND_BLOCK_N == plans.XBAND_BN
    assert (
        loader.xband_variant() == "xb" and loader.xband_variant("ow256") == "ow256_xb"
    )
    assert (
        loader.xband_variant("bulk_ow256") == "ow256_xb"
        and loader.xband_variant("bulk") == "xb"
    )
    # The fused multiply-add builds keep their trailing ``fma`` token after ``xb``.
    assert (
        loader.xband_variant("fma") == "xb_fma"
        and loader.xband_variant("ow256_fma") == "ow256_xb_fma"
    )
    assert loader.xband_variant("bulk_ow256_fma") == "ow256_xb_fma"
    for variant in loader.PRODUCER_VARIANTS:
        xb = loader.xband_variant(variant)
        staged = loader.XBAND_STAGED_VARIANT[variant]
        assert (
            "bulk" not in staged
            and loader.PRODUCER_VARIANT_OPERAND_LOAD_WIDTH[staged]
            == (loader.PRODUCER_VARIANT_OPERAND_LOAD_WIDTH[variant])
        )
        assert loader.is_xband_variant(xb) and not loader.is_xband_variant(variant)
        assert loader.base_variant(xb) == staged == loader.base_variant(staged)
        plain = loader.producer_key(256, True, xb)
        assert plain == f"producer_bn256_pair_{xb}"
        assert parse(plain) == (256, True, xb, None)
        for phase in loader.FOLD_PHASES:
            fold = loader.fold_producer_key(256, True, phase, xb)
            assert parse(fold) == (256, True, xb, phase)
        for block_n, pair in ((128, True), (256, False), (128, False)):
            with pytest.raises(ValueError, match="pair producer"):
                loader.producer_key(block_n, pair, xb)
    for bad in (
        "producer_bn128_pair_ow256_xb",
        "producer_bn256_ow256_xb",
        "producer_bn256_pair_xb_ow256",
        "producer_bn256_pair_ow512_xb",
    ):
        with pytest.raises(ValueError):
            parse(bad)
    with pytest.raises(ValueError, match="already"):
        loader.xband_variant("ow256_xb")


def test_producer_variant_default_table():
    """The default producer build per architecture and world size (the table the exporter renders), and its
    delivery: every default build is delivered with both rendezvous phases for at least one tile width, every
    variant program has its baseline twin, and ``delivered_producer_variants`` reports exactly the route table."""

    if not loader.ROUTES:
        pytest.skip("program tables are filled by the exporter")
    table = loader.PRODUCER_VARIANT_DEFAULT_BY_ARCH_WORLD
    assert table == {
        # SM90: the build whose epilogue stages each head for three storer warps
        "sm_90a": {2: "sw3", 4: "sw3", 8: "sw3"},
        # SM100 / SM103: the fused multiply-add epilogue builds
        "sm_100a": {2: "fma", 4: "ow256_fma", 8: "bulk_ow256_fma"},
        "sm_103a": {2: "fma", 4: "ow256_fma", 8: "ow256_fma"},
        # SM107: groups of 4 ranks run the bulk-row-store build (measured there); 2 ranks the baseline build
        "sm_107a": {2: "", 4: "bulk", 8: ""},
    }
    assert set(table) == set(loader.ARCH_FLAGS) == set(loader.ROUTES)
    for arch, row in table.items():
        assert tuple(row) == loader.SUPPORTED_WORLD_SIZES
        for world_size, variant in row.items():
            assert loader.default_producer_variant(arch, world_size) == variant
            assert variant in loader.PRODUCER_VARIANTS
            delivered = {
                (block_n, pair)
                for block_n in plans.BLOCK_N_CHOICES
                for pair in (False, True)
                if loader.has_producer(arch, block_n, pair, variant)
            }
            assert delivered, (arch, world_size, variant)
            for block_n, pair in delivered:
                assert loader.has_fold_producer(arch, block_n, pair, variant)
                assert loader.has_producer(arch, block_n, pair)
                assert loader.has_fold_producer(arch, block_n, pair)
                assert variant in loader.delivered_producer_variants(
                    arch, block_n, pair
                )
            # The cross-destination-band build of the default build, with both rendezvous phases, wherever its
            # gate ships on by default.
            if plans.default_xband_min_m(arch, world_size=world_size):
                assert loader.has_xband_producer(
                    arch, variant, fold=loader.default_fold_barrier(arch)
                )
    # Every delivered cross-destination-band build extends a build of its cell's default.
    for arch, routes in loader.ROUTES.items():
        xb_variants = {
            loader.parse_producer_key(key).variant
            for key in routes
            if key.startswith("producer_bn")
            and loader.is_xband_variant(loader.parse_producer_key(key).variant)
        }
        assert xb_variants == {
            loader.xband_variant(table[arch][world_size])
            for world_size in loader.SUPPORTED_WORLD_SIZES
            if plans.default_xband_min_m(arch, world_size=world_size)
        }
    with pytest.raises(ValueError, match="no default producer build"):
        loader.default_producer_variant("sm_120a", 8)
    with pytest.raises(ValueError, match="world_size=16"):
        loader.default_producer_variant("sm_100a", 16)
    for arch, routes in loader.ROUTES.items():
        for key in routes:
            if not key.startswith("producer_bn"):
                continue
            parsed = loader.parse_producer_key(key)
            assert loader.has_producer(arch, parsed.block_n, parsed.pair)
            assert loader.has_fold_producer(arch, parsed.block_n, parsed.pair)
            assert loader.has_producer(
                arch, parsed.block_n, parsed.pair, parsed.variant
            )
            assert loader.has_fold_producer(
                arch, parsed.block_n, parsed.pair, parsed.variant
            )
        for block_n in plans.BLOCK_N_CHOICES:
            for pair in (False, True):
                delivered = loader.delivered_producer_variants(arch, block_n, pair)
                assert delivered == tuple(
                    variant
                    for variant in loader.PRODUCER_VARIANTS
                    if loader.has_producer(arch, block_n, pair, variant)
                )
                assert not delivered or delivered[0] == loader.SHIPPED_VARIANT
        if arch == "sm_90a":
            assert all(
                loader.parse_producer_key(key).variant in ("", "sw3")
                for key in routes
                if key.startswith("producer_bn")
            )


def test_producer_shipped_rows_table():
    """The rows that keep the baseline build under the default build
    (``PRODUCER_SHIPPED_ROWS_BY_ARCH_WORLD``: SM90 with 8 ranks at hidden 6144 and above from 8192 rows on), the
    session's per-row build (the baseline build on such a row under the default build only) and its delivery: the
    baseline build of every SM90 tile width that delivers the default build, with both rendezvous phases."""

    assert plans.PRODUCER_SHIPPED_ROWS_BY_ARCH_WORLD == {("sm_90a", 8): (6144, 8192)}
    gated = {
        (hidden, rows)
        for hidden in (3072, 4096, 6144, 8192)
        for rows in (4096, 8191, 8192, 16384)
        if plans.default_shipped_producer_row("sm_90a", 8, hidden, rows)
    }
    assert gated == {(6144, 8192), (6144, 16384), (8192, 8192), (8192, 16384)}
    for arch in ("sm_90a", "sm_100a", "sm_103a", "sm_107a"):
        for world_size in plans.SUPPORTED_WORLD_SIZES:
            if (arch, world_size) != ("sm_90a", 8):
                assert not plans.default_shipped_producer_row(
                    arch, world_size, 8192, 65536
                )
    session = backend.CakeQkvExchangeSession.__new__(backend.CakeQkvExchangeSession)
    session.arch, session.world_size, session.hidden = "sm_90a", 8, 8192
    session.producer_variant, session.producer_variant_source = "sw3", "default"
    assert session._row_variant(16384) == loader.SHIPPED_VARIANT
    assert session._row_variant(4096) == "sw3"
    session.producer_variant_source = "env"
    assert session._row_variant(16384) == "sw3"
    if not loader.ROUTES:
        return
    for block_n in plans.BLOCK_N_CHOICES:
        for pair in (False, True):
            if loader.has_producer("sm_90a", block_n, pair, "sw3"):
                assert loader.has_producer("sm_90a", block_n, pair)
                assert loader.has_fold_producer("sm_90a", block_n, pair)


def test_producer_variant_resolution(monkeypatch):
    """Environment over the (architecture, world size) default: ``auto`` keeps the table, ``shipped`` selects the
    baseline build, a delivered name selects that build; an undelivered build named by the environment fails closed
    naming the environment variable, the ``shipped`` opt-out and the delivered builds.  A default build that is
    delivered for another tile width only yields to the baseline build of this one; a default build delivered for
    no tile width (a table without the variant programs) fails closed."""

    if not loader.ROUTES:
        pytest.skip("program tables are filled by the exporter")
    env = backend.PRODUCER_VARIANT_ENV
    resolve = backend.resolve_producer_variant
    for arch in loader.ROUTES:
        for block_n in plans.BLOCK_N_CHOICES:
            delivered = {
                variant
                for pair in (False, True)
                for variant in loader.delivered_producer_variants(arch, block_n, pair)
            }
            if not delivered:
                continue
            for world_size in loader.SUPPORTED_WORLD_SIZES:
                default = loader.default_producer_variant(arch, world_size)
                delivered_elsewhere = any(
                    loader.has_producer(arch, other, pair, default)
                    for other in plans.BLOCK_N_CHOICES
                    for pair in (False, True)
                )
                for value in (None, "auto", ""):
                    if value is None:
                        monkeypatch.delenv(env, raising=False)
                    else:
                        monkeypatch.setenv(env, value)
                    if default in delivered:
                        resolved = resolve(arch, world_size, block_n)
                        assert resolved == (default, "default")
                    elif delivered_elsewhere and "" in delivered:
                        resolved = resolve(arch, world_size, block_n)
                        assert resolved == ("", "default")
                    else:
                        with pytest.raises(ValueError, match=f"{env}=shipped"):
                            resolve(arch, world_size, block_n)
                monkeypatch.setenv(env, "shipped")
                assert resolve(arch, world_size, block_n) == ("", "env")
                for name in loader.PRODUCER_VARIANTS[1:]:
                    monkeypatch.setenv(env, name)
                    if name in delivered:
                        assert resolve(arch, world_size, block_n) == (name, "env")
                    else:
                        with pytest.raises(ValueError, match="delivered") as info:
                            resolve(arch, world_size, block_n)
                        assert env in str(info.value) and "shipped" in str(info.value)
                for bad in ("wide", "ow128", "1"):
                    monkeypatch.setenv(env, bad)
                    with pytest.raises(ValueError, match=env):
                        resolve(arch, world_size, block_n)
    monkeypatch.delenv(env, raising=False)
    assert backend.producer_variant_from_env() is None
    with pytest.raises(ValueError, match="world_size=16"):
        resolve("sm_100a", 16, 128)
    # A route table without the variant programs (a patched table): the default fails closed naming the opt-out
    # wherever the default is a variant; ``shipped`` resolves.
    for arch in loader.ROUTES:
        stripped = {
            key: value
            for key, value in loader.ROUTES[arch].items()
            if not key.startswith("producer_bn")
            or loader.parse_producer_key(key).variant == ""
        }
        with monkeypatch.context() as patched:
            patched.setattr(loader, "ROUTES", {**loader.ROUTES, arch: stripped})
            for block_n in plans.BLOCK_N_CHOICES:
                if not any(
                    loader.has_producer(arch, block_n, pair) for pair in (False, True)
                ):
                    continue
                for world_size in loader.SUPPORTED_WORLD_SIZES:
                    default = loader.default_producer_variant(arch, world_size)
                    patched.delenv(env, raising=False)
                    if default:
                        with pytest.raises(ValueError, match=f"{env}=shipped"):
                            resolve(arch, world_size, block_n)
                    else:
                        assert resolve(arch, world_size, block_n) == ("", "default")
                    patched.setenv(env, "shipped")
                    assert resolve(arch, world_size, block_n) == ("", "env")


def test_default_producer_variant_resolves_at_every_selectable_tile_width(monkeypatch):
    """Every (architecture, world size) default resolves at every tile width ``block_n_for`` selects for a
    supported geometry (1 to 16 heads per rank): to the default build where the route table delivers it for that
    width, otherwise to the baseline build of the width, delivered with both rendezvous phases for every pair flag
    the baseline build is delivered for.  The fallback cells are the ``block_n=128`` cells of the two-rank ``fma``
    defaults and of the SM100 four-rank ``ow256_fma`` default (hidden 768 with 6 heads over 2 ranks is one)."""

    if not loader.ROUTES:
        pytest.skip("program tables are filled by the exporter")
    monkeypatch.delenv(backend.PRODUCER_VARIANT_ENV, raising=False)
    odd_band = plans.Geometry(world_size=2, batch=1, s_local=4096, hidden=768)
    assert plans.block_n_for(odd_band) == 128
    fallbacks = set()
    for arch in loader.ROUTES:
        for world_size in loader.supported_world_sizes(arch):
            default = loader.default_producer_variant(arch, world_size)
            widths = {
                plans.block_n_for(
                    plans.Geometry(
                        world_size=world_size,
                        batch=1,
                        s_local=128,
                        hidden=world_size * heads_per_rank * HEAD_DIM,
                    )
                )
                for heads_per_rank in range(1, 17)
            }
            assert widths == set(plans.BLOCK_N_CHOICES)
            for block_n in sorted(widths):
                pairs = [
                    pair
                    for pair in (False, True)
                    if loader.has_producer(arch, block_n, pair)
                ]
                if not pairs:
                    # No build of this tile width on the architecture: every session of it refuses.
                    continue
                variant, source = backend.resolve_producer_variant(
                    arch, world_size, block_n
                )
                assert source == "default"
                here = any(
                    loader.has_producer(arch, block_n, pair, default)
                    for pair in (False, True)
                )
                assert variant == (default if here else loader.SHIPPED_VARIANT)
                if variant != default:
                    fallbacks.add((arch, world_size, block_n))
                for pair in pairs:
                    assert loader.has_producer(arch, block_n, pair, variant)
                    assert loader.has_fold_producer(arch, block_n, pair, variant)
    assert fallbacks == {
        ("sm_100a", 2, 128),
        ("sm_100a", 4, 128),
        ("sm_103a", 2, 128),
    }
    assert backend.resolve_producer_variant("sm_100a", 2, 128) == ("", "default")
    assert backend.resolve_producer_variant("sm_103a", 2, 128) == ("", "default")


def test_validate_refuses_misaligned_cos_sin_and_norm_weights():
    """``CakeQkvExchangeSession._validate`` refuses a ``cos_sin`` or norm-weight view that does not start at a
    16-byte-aligned address for every producer build, and one that does not start at a 32-byte-aligned address for
    the wide-load builds (naming the opt-out); aligned operands pass.  Checked on CPU tensors through a session
    stub: ``_validate`` reads only the session's device, geometry and producer build."""

    session = backend.CakeQkvExchangeSession.__new__(backend.CakeQkvExchangeSession)
    session.device = torch.device("cpu")
    session.hidden, session.head_dim, session.max_tokens = 256, HEAD_DIM, 64
    batch, s_local = 1, 16

    def placed(shape, dtype, offset_bytes):
        """A contiguous view of ``shape`` that starts ``offset_bytes`` past a 64-byte boundary."""

        itemsize = torch.empty((), dtype=dtype).element_size()
        numel = math.prod(shape)
        buffer = torch.zeros(numel + 128 // itemsize, dtype=dtype)
        start = (-buffer.data_ptr() % 64 + offset_bytes) // itemsize
        view = buffer[start : start + numel].view(shape)
        assert view.is_contiguous() and view.data_ptr() % 64 == offset_bytes
        return view

    x = torch.zeros((batch, s_local, session.hidden), dtype=torch.bfloat16)
    w_qkv = torch.zeros((3 * session.hidden, session.hidden), dtype=torch.bfloat16)
    operands = {
        "q_norm_weight": ((HEAD_DIM,), torch.bfloat16),
        "k_norm_weight": ((HEAD_DIM,), torch.bfloat16),
        "cos_sin": ((batch * s_local, HEAD_DIM), torch.float32),
    }

    def call(offsets):
        q_scale, k_scale, cos_sin = (
            placed(*operands[name], offsets.get(name, 0)) for name in operands
        )
        return session._validate(x, w_qkv, q_scale, k_scale, cos_sin)

    for variant in loader.PRODUCER_VARIANTS:
        session.producer_variant = variant
        wide = loader.PRODUCER_VARIANT_OPERAND_LOAD_WIDTH[variant] == 256
        assert call({}) == (batch, s_local)
        for name, (_shape, dtype) in operands.items():
            itemsize = torch.empty((), dtype=dtype).element_size()
            with pytest.raises(ValueError, match=f"{name} must start at a 16-byte"):
                call({name: itemsize})
            with pytest.raises(ValueError, match=f"{name} must start at a 16-byte"):
                call({name: 8})
            if wide:
                with pytest.raises(
                    ValueError, match=f"{backend.PRODUCER_VARIANT_ENV}=shipped"
                ):
                    call({name: 16})
            else:
                assert call({name: 16}) == (batch, s_local)


def test_dest_parts_cover_the_table_once_per_destination_run():
    """The launches of a ``per_dest`` call: contiguous destination runs that cover the tile table exactly once; pad
    rows of a pair plan stay with the run they pad."""

    geometry = plans.Geometry(world_size=2, batch=1, s_local=16384, hidden=1024)
    for arch in ("sm_100a", "sm_90a"):
        _policy, plan = plans.plan_for(
            geometry, "bshd", 1, arch=arch, pair_available=arch in plans.PAIR_ARCHES
        )
        parts = plans.dest_parts(plan)
        assert parts[0][0] == 0 and parts[-1][1] == plan.num_tiles
        assert all(stop > start for start, stop in parts)
        assert all(parts[i][1] == parts[i + 1][0] for i in range(len(parts) - 1))
        assert len(parts) >= geometry.world_size
        dest_column = plans.TILE_FIELDS.index("dest")
        for start, stop in parts:
            dests = {int(d) for d in plan.tiles[start:stop, dest_column].tolist()} - {
                -1
            }
            assert len(dests) == 1
        # The verdict of this row on each architecture (at its world size).
        verdict = plans.gate_rows(
            plans.default_shape_gates(arch, world_size=geometry.world_size),
            geometry.rows,
        )
        assert verdict.producer_launches == (
            "per_dest" if arch in ("sm_100a", "sm_103a") else "persistent"
        )
        assert verdict.words_capacity_per_dest is None


def test_loader_tables_are_consistent():
    if not loader.ROUTES:
        pytest.skip("program tables are filled by the exporter")
    for arch, table in loader.ROUTES.items():
        assert arch in loader.ARCH_FLAGS
        assert set(loader.STAGE_KEYS) <= set(table)
        for key, program in table.items():
            record = loader.PROGRAMS[program]
            assert arch in record["arches"]
            assert len(record["sources"]) == 2
            if key.startswith("producer_bn"):
                parsed = loader.parse_producer_key(key)
                assert ("_pair" in key) is parsed.pair
                assert record["cluster"] == ([2, 1, 1] if parsed.pair else [1, 1, 1])
                assert set(record["args"]) >= {"A", "B", "tiles", "ready_target"}
                if parsed.fold_phase is not None:
                    # The folded producer: the plain producer's ABI plus the rendezvous process-group
                    # parameters before the grid; the rendezvous warp is one extra warp per CTA on
                    # SM100 / SM103 and the otherwise idle TMA warp on SM90 (same thread count).
                    plain_key = key.rsplit("_fold", 1)[0]
                    assert plain_key == loader.producer_key(
                        parsed.block_n, parsed.pair, parsed.variant
                    )
                    plain = loader.PROGRAMS[table[plain_key]]
                    assert record["args"] == (
                        plain["args"][:-3]
                        + ["pg_world", "pg_rank", "pg_flags"]
                        + plain["args"][-3:]
                    )
                    extra_threads = 0 if arch == "sm_90a" else 32
                    assert record["block"] == [plain["block"][0] + extra_threads, 1, 1]
                    assert record["cluster"] == plain["cluster"]
                else:
                    assert "pg_flags" not in record["args"]
                    if loader.is_xband_variant(parsed.variant):
                        # A cross-destination-band build: the 256-column pair producer of a delivered build with
                        # that build's ABI and cluster (its tiles carry two more columns).
                        assert (parsed.block_n, parsed.pair) == (
                            loader.XBAND_BLOCK_N,
                            True,
                        )
                        base_key = loader.producer_key(
                            parsed.block_n, True, loader.base_variant(parsed.variant)
                        )
                        assert base_key in table and table[base_key] != program
                        base = loader.PROGRAMS[table[base_key]]
                        assert record["args"] == base["args"]
                        assert record["cluster"] == base["cluster"] == [2, 1, 1]
                    elif parsed.variant:
                        # A variant build: another program with the baseline build's ABI, block and cluster;
                        # the baseline build of the same tile width is delivered beside it (the opt-out).
                        baseline_key = loader.producer_key(parsed.block_n, parsed.pair)
                        assert baseline_key in table and table[baseline_key] != program
                        baseline = loader.PROGRAMS[table[baseline_key]]
                        assert record["args"] == baseline["args"]
                        assert record["block"] == baseline["block"]
                        assert record["cluster"] == baseline["cluster"]
            elif key.startswith("barrier_p"):
                assert record["args"][0] == "device_index"
            else:
                assert record["args"][0] == "ready"
        # Folded producers: both rendezvous phases of every delivered producer build where the
        # architecture folds by default (every delivered architecture, SM107
        # included); the ``else`` branch covers a table whose default is the barrier kernel.
        plain_keys = {
            key for key in table if key.startswith("producer_bn") and "_fold" not in key
        }
        complete = {
            f"{key}_fold{phase}" for key in plain_keys for phase in loader.FOLD_PHASES
        }
        fold_keys = {key for key in table if "_fold" in key}
        if loader.default_fold_barrier(arch):
            assert fold_keys == complete and loader.fold_producers_delivered(arch)
        else:
            assert not fold_keys and not loader.fold_producers_delivered(arch)
        # The policy never selects a producer the architecture does not deliver -- in the default build of the
        # row's world size (for the world sizes the architecture supports), with the baseline build delivered
        # beside a variant.
        for hidden, world_size, rows in (
            (3072, 8, 512),
            (3072, 8, 8192),
            (4096, 8, 512),
            (3072, 4, 1024),
            (4096, 4, 512),
            (8192, 2, 65536),
        ):
            if world_size not in loader.supported_world_sizes(arch):
                continue
            geometry = plans.Geometry(
                world_size=world_size, batch=1, s_local=rows, hidden=hidden
            )
            variant = loader.default_producer_variant(arch, world_size)
            policy = plans.plan_policy(
                geometry,
                arch=arch,
                pair_available=loader.has_producer(
                    arch, plans.block_n_for(geometry), True, variant
                ),
            )
            assert loader.has_producer(arch, policy.bn, policy.pair, variant)
            assert loader.has_fold_producer(
                arch, policy.bn, policy.pair, variant
            ) == loader.fold_producers_delivered(arch)
            assert loader.has_producer(arch, policy.bn, policy.pair)
            assert loader.has_fold_producer(arch, policy.bn, policy.pair)
            # The row's plan (with the 128-column tile table) selects a delivered producer too.
            row_policy, _plan = plans.plan_for(
                geometry,
                "bshd",
                0,
                arch=arch,
                pair_available=loader.has_producer(
                    arch, plans.block_n_for(geometry), True, variant
                ),
            )
            assert loader.has_producer(arch, row_policy.bn, row_policy.pair, variant)
            assert loader.has_fold_producer(
                arch, row_policy.bn, row_policy.pair, variant
            ) == loader.fold_producers_delivered(arch)


# --------------------------------------------------------------------------- multi-GPU


def _reference_local(x, w_qkv, q_scale, k_scale, cos_sin):
    """fp32 projection -> per-head RMSNorm (cast, then scale) -> interleaved RoPE -> BF16 ``(q, k, v)``."""

    batch, s_local, hidden = x.shape
    rows = batch * s_local
    heads = hidden // HEAD_DIM
    acc = (x.reshape(rows, hidden).float() @ w_qkv.float().t()).view(
        rows, 3, heads, HEAD_DIM
    )
    cos = cos_sin[:, : HEAD_DIM // 2][:, None, :]
    sin = cos_sin[:, HEAD_DIM // 2 :][:, None, :]

    def norm_rope(part, scale):
        rrms = torch.rsqrt(part.pow(2).mean(dim=-1, keepdim=True) + EPS)
        scaled = ((part * rrms).to(torch.bfloat16) * scale.to(torch.bfloat16)).float()
        rotated = torch.empty_like(scaled)
        rotated[..., 0::2] = scaled[..., 0::2] * cos - scaled[..., 1::2] * sin
        rotated[..., 1::2] = scaled[..., 1::2] * cos + scaled[..., 0::2] * sin
        return rotated.to(torch.bfloat16)

    q = norm_rope(acc[:, 0], q_scale)
    k = norm_rope(acc[:, 1], k_scale)
    v = acc[:, 2].to(torch.bfloat16)
    shape = (batch, s_local, heads, HEAD_DIM)
    return q.view(shape), k.view(shape), v.view(shape)


def _reference_per_operator(x, w_qkv, q_scale, k_scale, cos_sin):
    """The rounding points of the per-operator path (``functional.linear`` -> ``rmsnorm`` -> RoPE) in fp32 host math:

    1. the projection accumulates in fp32 and is stored in BF16 (cuBLAS with BF16 operands); ``v`` is that value;
    2. ``rmsnorm`` reads the BF16 projection, forms ``rsqrt(mean(x * x) + eps)`` in fp32, multiplies ``x * rrms``
       first and the fp32 weight second, and stores BF16 once;
    3. RoPE (``is_neox=False``) reads the BF16 normed values and the fp32 cos / sin table, rotates each interleaved
       pair in fp32 (``x0 * cos - x1 * sin``, ``x1 * cos + x0 * sin``) and stores BF16.

    Not reproduced: the fp32 accumulation order of the projection and of the sum of squares, the approximate
    ``rsqrt`` and the fused multiply-adds of the kernels.  They move a value across a BF16 rounding boundary only
    occasionally (about 2e-4 of the stored projection values between two fp32 accumulation orders on the host at
    hidden 1024 to 8192); at a near-zero RoPE output such a flip of an O(1) normed input is an absolute difference
    of a few BF16 spacings of the pair norm (0.03125 at most recorded for the per-operator path against this
    reference), inside the tolerance of ``_check``."""

    batch, s_local, hidden = x.shape
    rows = batch * s_local
    heads = hidden // HEAD_DIM
    acc = x.reshape(rows, hidden).float() @ w_qkv.float().t()
    qkv = acc.to(torch.bfloat16).view(rows, 3, heads, HEAD_DIM)
    cos = cos_sin[:, : HEAD_DIM // 2][:, None, :]
    sin = cos_sin[:, HEAD_DIM // 2 :][:, None, :]

    def norm_rope(part, scale):
        normed = part.float()
        rrms = torch.rsqrt(normed.pow(2).mean(dim=-1, keepdim=True) + EPS)
        normed = (normed * rrms * scale.float()).to(torch.bfloat16).float()
        rotated = torch.empty_like(normed)
        rotated[..., 0::2] = normed[..., 0::2] * cos - normed[..., 1::2] * sin
        rotated[..., 1::2] = normed[..., 1::2] * cos + normed[..., 0::2] * sin
        return rotated.to(torch.bfloat16)

    shape = (batch, s_local, heads, HEAD_DIM)
    q = norm_rope(qkv[:, 0], q_scale).view(shape)
    k = norm_rope(qkv[:, 1], k_scale).view(shape)
    return q, k, qkv[:, 2].reshape(shape)


def _redistribute(local, group, world_size, rank):
    """``[B, S_local, H, D] -> [B, S_global, H_local, D]``: this rank's head band of every rank's tokens."""

    peers = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(peers, local.contiguous(), group=group)
    h_local = local.shape[2] // world_size
    return torch.cat(
        [peer[:, :, rank * h_local : (rank + 1) * h_local, :] for peer in peers], dim=1
    )


def _expected(x, w_qkv, q_scale, k_scale, cos_sin, group, world_size, rank, ref=None):
    q, k, v = (ref or _reference_local)(x, w_qkv, q_scale, k_scale, cos_sin)
    return tuple(_redistribute(t, group, world_size, rank) for t in (q, k, v))


def _check(actual, expected):
    """A device output against its host fp32 reference: finite, and ``assert_close`` at ``atol = 0.05``,
    ``rtol = 1e-2``.  The absolute tolerance covers BF16 rounding flips between the device and the host: both round
    the normed values to BF16 at the same points but not always to the same side (fp32 accumulation order,
    approximate ``rsqrt``, fused multiply-adds), and at a near-zero RoPE output -- two terms of magnitude up to ~4
    cancelling -- one such flip of a stored input is an absolute difference of a few BF16 spacings of the pair norm.
    Recorded on two-, four- and eight-GPU runs of three architectures: the largest difference of any element is
    0.03125, both for the fused path and for the per-operator path against its mirrored reference (the largest
    element outside ``0.01 + 0.01 * |expected|`` was 0.0137, fused path); the tolerance is set above these maxima,
    not derived as a bound.  ``v`` (the BF16 projection, no cancellation; at most 0.015625 recorded) shares the
    tolerance for uniformity.  What the tolerance rejects: an element with |diff| > 0.05 + 0.01 * |expected| -- a
    wrong scale, pairing, table row, slot or transport moves whole O(1) values on many elements.  What it admits:
    any per-element error inside that bound (0.06 at unit scale), a uniform relative error of the q / k outputs
    below ~2.2 %, a RoPE angle error below ~0.012 rad.  The bitwise comparisons (replayed first against second
    captured call, kept slot views, folded against barrier launch path) cover the replay, the transport and the
    launch paths exactly."""

    assert torch.isfinite(actual).all(), "non-finite elements in the output"
    torch.testing.assert_close(actual.float(), expected.float(), atol=0.05, rtol=1e-2)


def _check_outputs(session, outputs, expected):
    if session.layout == "bshd":
        for actual, want in zip(outputs, expected, strict=True):
            _check(actual, want)
    else:
        for index, want in enumerate(expected):
            _check(outputs[..., index * HEAD_DIM : (index + 1) * HEAD_DIM], want)


def _check_per_operator_outputs(outputs, expected, group, device, rank, label):
    """``(q, k, v)`` of the per-operator path (``bshd`` layout) against the redistributed per-operator reference,
    with ``_check``.  The ranks agree on the tolerance verdict (the ``AssertionError`` of ``_check``) before any of
    them raises (a rank that raised alone would leave its peers blocked in the next collective); the message names
    the rank and the failing tensors."""

    failures = {}
    for name, actual, want in zip(("q", "k", "v"), outputs, expected, strict=True):
        try:
            _check(actual, want)
        except AssertionError as error:
            failures[name] = " ".join(str(error).split())
    failed = torch.tensor([int(bool(failures))], dtype=torch.int32, device=device)
    dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=group)
    assert int(failed.item()) == 0, (
        f"{label}: the per-operator path fails its reference on at least one rank; "
        f"rank {rank}: {failures or 'every tensor passed on this rank'}"
    )


def _operands(device, *, batch, s_local, hidden, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(
        (batch, s_local, hidden),
        dtype=torch.float32,
        device=device,
        generator=generator,
    ).to(torch.bfloat16)
    w_qkv = (
        torch.randn(
            (3 * hidden, hidden),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        * hidden**-0.5
    ).to(torch.bfloat16)
    q_scale = (
        1.0
        + 0.1
        * torch.randn(
            (HEAD_DIM,), dtype=torch.float32, device=device, generator=generator
        )
    ).to(torch.bfloat16)
    k_scale = (
        1.0
        + 0.1
        * torch.randn(
            (HEAD_DIM,), dtype=torch.float32, device=device, generator=generator
        )
    ).to(torch.bfloat16)
    positions = torch.arange(batch * s_local, device=device, dtype=torch.float64)
    omega = 1.0 / (
        10000.0
        ** (torch.arange(0, HEAD_DIM, 2, device=device, dtype=torch.float64) / HEAD_DIM)
    )
    angle = positions[:, None] * omega[None, :]
    cos_sin = (
        torch.cat([torch.cos(angle), torch.sin(angle)], dim=-1).float().contiguous()
    )
    return x, w_qkv, q_scale, k_scale, cos_sin


def _run_worker(rank, world_size, port):
    from flashinfer.comm import QkvNormRopeExchangeOverlap

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    group = dist.group.WORLD
    hidden = 1024
    heads = hidden // HEAD_DIM
    try:
        # 1. Tail row count (S_local not a multiple of the 128-row tile), both layouts, eager calls.
        batch, s_local = 1, 200
        x, w_qkv, q_scale, k_scale, cos_sin = _operands(
            device, batch=batch, s_local=s_local, hidden=hidden, seed=11 + rank
        )
        expected = _expected(
            x, w_qkv, q_scale, k_scale, cos_sin, group, world_size, rank
        )
        for layout in ("bshd", "fused"):
            session = QkvNormRopeExchangeOverlap(
                group,
                hidden=hidden,
                num_heads=heads,
                layout=layout,
                max_tokens=batch * s_local,
                backend="nvlink",
                device=device,
            )
            assert session.backend == "nvlink" and session.graph_capturable
            try:
                for _call in range(3):
                    _check_outputs(
                        session, session(x, w_qkv, q_scale, k_scale, cos_sin), expected
                    )
                owned = (
                    tuple(torch.empty_like(t) for t in expected)
                    if layout == "bshd"
                    else torch.empty(
                        expected[0].shape[:-1] + (3 * HEAD_DIM,),
                        dtype=torch.bfloat16,
                        device=device,
                    )
                )
                _check_outputs(
                    session,
                    session(x, w_qkv, q_scale, k_scale, cos_sin, out=owned),
                    expected,
                )
                with pytest.raises(ValueError, match="max_tokens"):
                    session(
                        torch.randn(
                            (1, s_local + 8, hidden),
                            dtype=torch.bfloat16,
                            device=device,
                        ),
                        w_qkv,
                        q_scale,
                        k_scale,
                        torch.zeros(
                            (s_local + 8, HEAD_DIM), dtype=torch.float32, device=device
                        ),
                    )
            finally:
                session.close()

        # 2. Capture / replay x 3 with fresh inputs, eager calls between replays; refusals under capture.
        batch, s_local = 2, 256
        x, w_qkv, q_scale, k_scale, cos_sin = _operands(
            device, batch=batch, s_local=s_local, hidden=hidden, seed=101 + rank
        )
        session = QkvNormRopeExchangeOverlap(
            group,
            hidden=hidden,
            num_heads=heads,
            layout="bshd",
            max_tokens=batch * s_local,
            backend="nvlink",
            device=device,
        )
        try:
            # One eager call per slot prepares the row and warms both slots.
            for _call in range(session.num_slots):
                session(x, w_qkv, q_scale, k_scale, cos_sin)
            capture_stream = torch.cuda.Stream(device=device)
            capture_stream.wait_stream(torch.cuda.current_stream(device))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                first = session(x, w_qkv, q_scale, k_scale, cos_sin)
                second = session(x, w_qkv, q_scale, k_scale, cos_sin)
            # Fresh inputs from a seeded generator, so that a miss of the fused-path check below is
            # reproducible (the inputs of every other part are seeded too).
            fresh = torch.Generator(device=device).manual_seed(201 + rank)
            eager_x = torch.randn(
                x.shape, dtype=torch.float32, device=device, generator=fresh
            ).to(torch.bfloat16)
            for _replay in range(3):
                x.copy_(
                    torch.randn(
                        x.shape, dtype=torch.float32, device=device, generator=fresh
                    ).to(torch.bfloat16)
                )
                dist.barrier(group=group)
                graph.replay()
                torch.cuda.synchronize(device)
                replay_expected = _expected(
                    x, w_qkv, q_scale, k_scale, cos_sin, group, world_size, rank
                )
                _check_outputs(session, first, replay_expected)
                _check_outputs(session, second, replay_expected)
                # Both captured calls read the same ``x`` through consecutive receive
                # slots and alternating fold phases; the kernel is deterministic, so
                # their replayed outputs must be bitwise identical.
                for one, other in zip(first, second, strict=True):
                    assert torch.equal(one, other), (
                        "replayed outputs of the two captured calls differ: both "
                        "read the same inputs through consecutive slots and "
                        "alternating fold phases, so they must be bitwise identical"
                    )
                eager_expected = _expected(
                    eager_x, w_qkv, q_scale, k_scale, cos_sin, group, world_size, rank
                )
                _check_outputs(
                    session,
                    session(eager_x, w_qkv, q_scale, k_scale, cos_sin),
                    eager_expected,
                )
            # A row count first seen inside a capture is refused (nothing may allocate under capture).
            small_x = torch.randn((1, 128, hidden), dtype=torch.bfloat16, device=device)
            small_cos = torch.zeros((128, HEAD_DIM), dtype=torch.float32, device=device)
            refusal = torch.cuda.CUDAGraph()
            with (
                pytest.raises(RuntimeError, match="inside a CUDA-graph capture"),
                torch.cuda.graph(refusal, stream=torch.cuda.Stream(device=device)),
            ):
                session(small_x, w_qkv, q_scale, k_scale, small_cos)
            del refusal
            # The session is still healthy after the refused capture: the eager path serves the new row count.
            small_expected = _expected(
                small_x, w_qkv, q_scale, k_scale, small_cos, group, world_size, rank
            )
            _check_outputs(
                session,
                session(small_x, w_qkv, q_scale, k_scale, small_cos),
                small_expected,
            )
            del graph, first, second
        finally:
            session.close()

        # 3. Two sessions in flight on one group (one per layout), calls alternating.
        a = QkvNormRopeExchangeOverlap(
            group,
            hidden=hidden,
            num_heads=heads,
            layout="bshd",
            max_tokens=batch * s_local,
            backend="nvlink",
            device=device,
        )
        b = QkvNormRopeExchangeOverlap(
            group,
            hidden=hidden,
            num_heads=heads,
            layout="fused",
            max_tokens=batch * s_local,
            backend="nvlink",
            device=device,
        )
        try:
            expected = _expected(
                x, w_qkv, q_scale, k_scale, cos_sin, group, world_size, rank
            )
            eager_expected = _expected(
                eager_x, w_qkv, q_scale, k_scale, cos_sin, group, world_size, rank
            )
            for _round in range(2):
                out_a = a(x, w_qkv, q_scale, k_scale, cos_sin)
                out_b = b(eager_x, w_qkv, q_scale, k_scale, cos_sin)
                _check_outputs(a, out_a, expected)
                _check_outputs(b, out_b, eager_expected)
        finally:
            a.close()
            b.close()

        # 4. NCCL path: the per-operator path (cuBLAS projection, ``rmsnorm``, RoPE) against the reference that
        #    mirrors its rounding points; not capturable.
        nccl = QkvNormRopeExchangeOverlap(
            group,
            hidden=hidden,
            num_heads=heads,
            layout="bshd",
            max_tokens=batch * s_local,
            backend="nccl",
            device=device,
        )
        try:
            assert nccl.backend == "nccl" and not nccl.graph_capturable
            per_operator = _expected(
                x,
                w_qkv,
                q_scale,
                k_scale,
                cos_sin,
                group,
                world_size,
                rank,
                ref=_reference_per_operator,
            )
            first = nccl(x, w_qkv, q_scale, k_scale, cos_sin)
            _check_per_operator_outputs(
                first, per_operator, group, device, rank, "part 4, first call"
            )
            # A call's outputs stay valid for num_slots - 1 further calls on this path too:
            # the next call lands in another receive slot and leaves the first views intact.
            kept = tuple(t.clone() for t in first)
            nccl(x.neg(), w_qkv, q_scale, k_scale, cos_sin)
            torch.cuda.synchronize(device)
            for before, view in zip(kept, first, strict=True):
                assert torch.equal(before, view)
            _check_per_operator_outputs(
                nccl(x, w_qkv, q_scale, k_scale, cos_sin),
                per_operator,
                group,
                device,
                rank,
                "part 4, third call",
            )
            refusal = torch.cuda.CUDAGraph()
            with (
                pytest.raises(RuntimeError, match="not CUDA-graph capturable"),
                torch.cuda.graph(refusal, stream=torch.cuda.Stream(device=device)),
            ):
                nccl(x, w_qkv, q_scale, k_scale, cos_sin)
            del refusal
        finally:
            nccl.close()

        # 5. Launch paths: the architecture default (the folded rendezvous on every delivered
        #    architecture, SM107 included) against the explicit barrier path --
        #    bitwise-equal outputs; two versus three launches per call where the default folds.
        arch = loader.device_facts(device.index).arch
        fold_default = loader.default_fold_barrier(arch)
        default_path = QkvNormRopeExchangeOverlap(
            group,
            hidden=hidden,
            num_heads=heads,
            layout="bshd",
            max_tokens=batch * s_local,
            backend="nvlink",
            device=device,
        )
        barrier_path = QkvNormRopeExchangeOverlap(
            group,
            hidden=hidden,
            num_heads=heads,
            layout="bshd",
            max_tokens=batch * s_local,
            backend="nvlink",
            device=device,
            fold_barrier=False,
        )
        try:
            default_info = default_path.describe()
            barrier_info = barrier_path.describe()
            assert default_info["fold_barrier"] is fold_default
            assert default_info["fold_barrier_requested"] == "auto"
            assert default_info["launches_per_call"] == (2 if fold_default else 3)
            assert default_info["launch_sequence"] == (
                ["producer_fold<phase>", "wait"]
                if fold_default
                else ["barrier_p<phase>", "producer", "wait"]
            )
            assert barrier_info["fold_barrier"] is False
            assert barrier_info["fold_barrier_requested"] is False
            assert barrier_info["launches_per_call"] == 3
            assert barrier_info["launch_sequence"] == [
                "barrier_p<phase>",
                "producer",
                "wait",
            ]
            # The producer build and the group scaling of the architecture default for this world size.
            variant = loader.default_producer_variant(arch, world_size)
            assert default_info["producer_variant"] == {
                "value": variant,
                "source": "default",
                "env": backend.PRODUCER_VARIANT_ENV,
                "default": variant,
            }
            assert default_info["group_m_scale"] == {
                "value": plans.default_group_m_scale(arch, world_size),
                "source": "default",
                "env": backend.GROUP_M_SCALE_ENV,
                "default": plans.default_group_m_scale(arch, world_size),
            }
            assert default_info["reserved_sms"] == plans.RESERVED_SMS
            bound = default_path.bound_programs(batch, s_local)
            assert bound == barrier_path.bound_programs(batch, s_local)
            assert "barrier_p0" in bound and "producer" in bound
            if fold_default:
                assert "producer_fold0" in bound and "producer_fold1" in bound
                # Six delivered stages on a baseline-build row, nine where the default build is a variant (the
                # baseline build's three programs ride along as the ``shipped`` opt-out).
                assert len(bound) == (6 if variant == "" else 9)
            else:
                assert not any("_fold" in key for key in bound) and len(bound) == 4
            block_n = default_info["block_n"]
            row_policy = plans.plan_policy(
                plans.Geometry(
                    world_size=world_size, batch=batch, s_local=s_local, hidden=hidden
                ),
                arch=arch,
                pair_available=default_info["pair_available"][block_n],
            )
            assert bound["producer"] == loader.producer_program(
                arch, block_n, row_policy.pair, variant
            )
            for phase in loader.FOLD_PHASES if fold_default else ():
                assert bound[f"producer_fold{phase}"] == loader.fold_producer_program(
                    arch, block_n, row_policy.pair, phase, variant
                )
            if variant:
                assert bound["producer_shipped"] == loader.producer_program(
                    arch, block_n, row_policy.pair
                )
                assert bound["producer_shipped"] != bound["producer"]
                for phase in loader.FOLD_PHASES if fold_default else ():
                    shipped_fold = loader.fold_producer_program(
                        arch, block_n, row_policy.pair, phase
                    )
                    assert bound[f"producer_shipped_fold{phase}"] == shipped_fold
            else:
                assert not any(name.startswith("producer_shipped") for name in bound)
            # The row-count gates of the architecture default on this row (``batch * s_local`` = 512 rows): the
            # resolved table, its source and the prepared row's verdict agree with the plan module.
            gates = default_path.describe()["shape_gates"]
            assert (
                gates["values"]
                == plans.spell_shape_gates(
                    plans.default_shape_gates(arch, world_size=world_size)
                )
                == gates["default"]
            )
            assert (
                gates["source"] == "default" and gates["env"] == backend.SHAPE_GATES_ENV
            )
            verdict = plans.gate_rows(gates["values"], batch * s_local, bn=block_n)
            row_plan = default_path.describe()["prepared_row_plans"][
                f"{batch}x{s_local}"
            ]
            assert row_plan["fired"] == list(verdict.fired)
            assert (
                row_plan["producer_launches"]
                == verdict.producer_launches
                == "persistent"
            )
            assert (
                row_plan["words_capacity_per_dest"] == verdict.words_capacity_per_dest
            )
            assert row_plan["reserved_sms"] is None and verdict.reserved_sms is None
            assert row_plan["producer_parts"] == 1 and row_plan[
                "launches_per_call"
            ] == (2 if fold_default else 3)
            assert (
                max(row_plan["part_grids"])
                <= default_info["sm_count"] - plans.RESERVED_SMS
            )
            if verdict.words_capacity_per_dest == 1:
                assert row_plan["num_words"] == 1
            for _round in range(3):
                from_default = tuple(
                    t.clone() for t in default_path(x, w_qkv, q_scale, k_scale, cos_sin)
                )
                from_barrier = barrier_path(x, w_qkv, q_scale, k_scale, cos_sin)
                torch.cuda.synchronize(device)
                _check_outputs(default_path, from_default, expected)
                for one, other in zip(from_default, from_barrier, strict=True):
                    assert torch.equal(one, other)
        finally:
            default_path.close()
            barrier_path.close()

        # 6. Heads not divisible by the world size raise on every rank before any collective.
        assert INDIVISIBLE_HEADS % world_size
        with pytest.raises(ValueError, match="divisible by world_size"):
            QkvNormRopeExchangeOverlap(
                group,
                hidden=INDIVISIBLE_HEADS * HEAD_DIM,
                num_heads=INDIVISIBLE_HEADS,
                layout="bshd",
                max_tokens=128,
                backend="nvlink",
                device=device,
            )
        torch.cuda.synchronize(device)
        dist.barrier(group=group)

        # 7. Row-count gates at the large-row threshold (16384 rows per rank): the per-destination launch path
        #    (the architecture default where the K table's threshold for this hidden size is 16384 -- SM100 with 8 or
        #    2 ranks --, the explicit environment override elsewhere, with the cross-destination-band gate off; on
        #    SM90 the default large-row reserved-SM gate fires beside it) against the persistent launch (every gate
        #    off) -- bitwise-equal outputs, one producer launch per destination run; on SM90 also against the
        #    explicit ``reserved_sms_large_m=0,words_capacity_small_m=0`` opt-out (the reserved-off session switches
        #    the word gate off too, so the 8-rank ``all`` default does not fire on this row and nothing fires on that
        #    session).
        batch, s_local = 1, 16384
        x, w_qkv, q_scale, k_scale, cos_sin = _operands(
            device, batch=batch, s_local=s_local, hidden=hidden, seed=31 + rank
        )
        expected = _expected(
            x, w_qkv, q_scale, k_scale, cos_sin, group, world_size, rank
        )
        default_gates = plans.default_shape_gates(arch, world_size=world_size)
        # The verdict is read with the row's tile width and hidden size, as the backend reads it: the pair-tile gate
        # of the (architecture, world size) table fires on 256-column rows only, the K table reads the hidden size,
        # and the cross-destination-band gate (where its build is delivered) takes precedence.
        geometry = plans.Geometry(
            world_size=world_size, batch=batch, s_local=s_local, hidden=hidden
        )
        block_n = plans.block_n_for(geometry)
        default_xband = plans.default_xband_min_m(arch, world_size=world_size)
        if not loader.has_xband_producer(
            arch, loader.default_producer_variant(arch, world_size), fold=fold_default
        ):
            default_xband = 0
        default_verdict = plans.gate_rows(
            default_gates,
            batch * s_local,
            bn=block_n,
            hidden=hidden,
            k_table=plans.default_shape_gate_k_table(arch, world_size=world_size),
            xband_min_m=default_xband,
            xband_eligible=plans.xband_eligible(geometry),
        )
        default_is_per_dest = default_verdict.producer_launches == "per_dest"
        # A fired pair-tile gate selects the single-CTA producer (no delivered cell ships the gate: the table
        # decides, the verdict reports it).
        pair_gate = default_gates["pair_off_large_m"]
        default_pair_off = bool(
            pair_gate
            and block_n == plans.SHAPE_GATE_BN["pair_off_large_m"]
            and batch * s_local >= pair_gate
        )
        assert (default_verdict.pair is False) is default_pair_off
        default_reserves_none = default_verdict.reserved_sms == 0
        gated_env = (
            None
            if default_is_per_dest
            else "per_dest_large_m=16384,words_capacity_small_m=0,xband_min_m=0"
        )

        def _session_with(env):
            """A session built with the given environment overrides (``name -> value``) set, then restored."""

            previous = {name: os.environ.pop(name, None) for name in env}
            os.environ.update(env)
            try:
                return QkvNormRopeExchangeOverlap(
                    group,
                    hidden=hidden,
                    num_heads=heads,
                    layout="bshd",
                    max_tokens=batch * s_local,
                    backend="nvlink",
                    device=device,
                )
            finally:
                for name, value in previous.items():
                    os.environ.pop(name, None)
                    if value is not None:
                        os.environ[name] = value

        per_dest_path = _session_with(
            {} if gated_env is None else {backend.SHAPE_GATES_ENV: gated_env}
        )
        persistent_path = _session_with({backend.SHAPE_GATES_ENV: "off"})
        reserved_off_path = (
            _session_with(
                {
                    backend.SHAPE_GATES_ENV: (
                        "reserved_sms_large_m=0,words_capacity_small_m=0"
                    )
                }
            )
            if default_reserves_none
            else None
        )
        try:
            gated_info = per_dest_path.describe()
            plain_info = persistent_path.describe()
            assert gated_info["shape_gates"]["source"] == (
                "default" if default_is_per_dest else "env"
            )
            assert plain_info["shape_gates"]["source"] == "env"
            assert plain_info["shape_gates"]["values"] == {
                name: plans.SHAPE_GATE_OFF for name in plans.SHAPE_GATE_NAMES
            }
            per_dest_path.bound_programs(batch, s_local)
            persistent_path.bound_programs(batch, s_local)
            gated_row = per_dest_path.describe()["prepared_row_plans"][
                f"{batch}x{s_local}"
            ]
            plain_row = persistent_path.describe()["prepared_row_plans"][
                f"{batch}x{s_local}"
            ]
            assert (
                gated_row["producer_launches"] == "per_dest"
                and "per_dest_large_m" in gated_row["fired"]
            )
            # The large-row reserved-SM gate keeps its architecture default beside the per-destination override:
            # it fires on SM90 (every SM runs the producer), stays off on SM100 / SM103; the pair-tile gate fires
            # beside the per-destination gate where the (architecture, world size) table says so (no delivered
            # cell does today; the default path -- ``gated_env`` None -- reads the table).
            expected_fired = (
                ["reserved_sms_large_m"] if default_reserves_none else []
            ) + ["per_dest_large_m"]
            if gated_env is None and default_pair_off:
                expected_fired.append("pair_off_large_m")
            assert gated_row["fired"] == expected_fired
            assert gated_row["pair"] == (
                default_verdict.pair if gated_env is None else None
            )
            assert gated_row["reserved_sms"] == (0 if default_reserves_none else None)
            assert (
                plain_row["producer_launches"] == "persistent"
                and plain_row["fired"] == []
            )
            assert plain_row["reserved_sms"] is None
            assert (
                max(plain_row["part_grids"])
                <= plain_info["sm_count"] - plans.RESERVED_SMS
            )
            if default_reserves_none:
                assert max(gated_row["part_grids"]) == gated_info["sm_count"]
            else:
                assert (
                    max(gated_row["part_grids"])
                    <= gated_info["sm_count"] - plans.RESERVED_SMS
                )
            assert block_n == gated_info["block_n"]
            _policy, plan = plans.plan_for(
                geometry,
                "bshd",
                rank,
                arch=arch,
                pair_available=gated_info["pair_available"][block_n],
                pair=default_verdict.pair if gated_env is None else None,
            )
            assert plan.pair is (
                False if (gated_env is None and default_pair_off) else _policy.pair
            )
            parts = plans.dest_parts(plan)
            assert gated_row["producer_parts"] == len(parts) >= world_size
            assert gated_row["launches_per_call"] == len(parts) + 1
            assert (
                plain_row["producer_parts"] == 1 and plain_row["launches_per_call"] == 2
            )
            assert gated_row["num_words"] == plain_row["num_words"], (
                "the large row keeps its readiness words"
            )
            if reserved_off_path is not None:
                reserved_off_path.bound_programs(batch, s_local)
                reserved_off_info = reserved_off_path.describe()
                reserved_off_row = reserved_off_info["prepared_row_plans"][
                    f"{batch}x{s_local}"
                ]
                reserved_off_gates = reserved_off_info["shape_gates"]
                assert reserved_off_gates["source"] == "env"
                assert reserved_off_gates["values"]["reserved_sms_large_m"] == 0
                assert reserved_off_gates["values"]["words_capacity_small_m"] == 0
                assert reserved_off_row["fired"] == []
                assert reserved_off_row["reserved_sms"] is None
                assert (
                    max(reserved_off_row["part_grids"])
                    <= reserved_off_info["sm_count"] - plans.RESERVED_SMS
                )
            for _round in range(3):
                from_per_dest = tuple(
                    t.clone()
                    for t in per_dest_path(x, w_qkv, q_scale, k_scale, cos_sin)
                )
                from_persistent = persistent_path(x, w_qkv, q_scale, k_scale, cos_sin)
                torch.cuda.synchronize(device)
                _check_outputs(per_dest_path, from_per_dest, expected)
                for one, other in zip(from_per_dest, from_persistent, strict=True):
                    assert torch.equal(one, other)
                if reserved_off_path is not None:
                    from_reserved_off = reserved_off_path(
                        x, w_qkv, q_scale, k_scale, cos_sin
                    )
                    torch.cuda.synchronize(device)
                    for one, other in zip(
                        from_per_dest, from_reserved_off, strict=True
                    ):
                        assert torch.equal(one, other)
        finally:
            per_dest_path.close()
            persistent_path.close()
            if reserved_off_path is not None:
                reserved_off_path.close()
        torch.cuda.synchronize(device)
        dist.barrier(group=group)

        # 8. The two opt-outs -- the baseline producer build (``FLASHINFER_QKV_EXCHANGE_PRODUCER_VARIANT=shipped``)
        #    and the baseline group scaling (``FLASHINFER_QKV_EXCHANGE_GROUP_M_SCALE=1``) -- against the
        #    (architecture, world size) defaults: bitwise-equal outputs; the bound producer programs differ exactly
        #    where the default build is a variant; a wide-load default (operand load width 256) additionally refuses a
        #    misaligned ``cos_sin`` view.
        batch, s_local = 1, 512
        x, w_qkv, q_scale, k_scale, cos_sin = _operands(
            device, batch=batch, s_local=s_local, hidden=hidden, seed=41 + rank
        )
        expected = _expected(
            x, w_qkv, q_scale, k_scale, cos_sin, group, world_size, rank
        )
        default_variant = loader.default_producer_variant(arch, world_size)
        default_scale = plans.default_group_m_scale(arch, world_size)
        default_session = _session_with({})
        opt_out_session = _session_with(
            {
                backend.PRODUCER_VARIANT_ENV: "shipped",
                backend.GROUP_M_SCALE_ENV: "1",
            }
        )
        try:
            default_info = default_session.describe()
            opt_out_info = opt_out_session.describe()
            assert default_info["producer_variant"]["value"] == default_variant
            assert default_info["producer_variant"]["source"] == "default"
            assert default_info["group_m_scale"]["value"] == default_scale
            assert default_info["group_m_scale"]["source"] == "default"
            assert opt_out_info["producer_variant"] == {
                "value": "",
                "source": "env",
                "env": backend.PRODUCER_VARIANT_ENV,
                "default": default_variant,
            }
            assert opt_out_info["group_m_scale"] == {
                "value": 1,
                "source": "env",
                "env": backend.GROUP_M_SCALE_ENV,
                "default": default_scale,
            }
            bound_default = default_session.bound_programs(batch, s_local)
            bound_opt_out = opt_out_session.bound_programs(batch, s_local)
            assert len(bound_opt_out) == 6
            assert len(bound_default) == (6 if default_variant == "" else 9)
            if default_variant:
                assert bound_default["producer"] != bound_opt_out["producer"]
                assert bound_default["producer_shipped"] == bound_opt_out["producer"]
                for phase in loader.FOLD_PHASES:
                    assert (
                        bound_default[f"producer_fold{phase}"]
                        != bound_opt_out[f"producer_fold{phase}"]
                    )
                    assert (
                        bound_default[f"producer_shipped_fold{phase}"]
                        == bound_opt_out[f"producer_fold{phase}"]
                    )
            else:
                assert bound_default == bound_opt_out
            default_row = default_session.describe()["prepared_row_plans"][
                f"{batch}x{s_local}"
            ]
            opt_out_row = opt_out_session.describe()["prepared_row_plans"][
                f"{batch}x{s_local}"
            ]
            assert default_row["num_words"] == opt_out_row["num_words"]
            for _round in range(3):
                from_default = tuple(
                    t.clone()
                    for t in default_session(x, w_qkv, q_scale, k_scale, cos_sin)
                )
                from_opt_out = opt_out_session(x, w_qkv, q_scale, k_scale, cos_sin)
                torch.cuda.synchronize(device)
                _check_outputs(default_session, from_default, expected)
                if default_variant.endswith("fma"):
                    # The fused multiply-add build rounds the norm and RoPE
                    # multiply-adds once, so its bits can differ from the
                    # baseline build's: both are checked against the reference.
                    _check_outputs(opt_out_session, from_opt_out, expected)
                else:
                    for one, other in zip(from_default, from_opt_out, strict=True):
                        assert torch.equal(one, other)
            if loader.PRODUCER_VARIANT_OPERAND_LOAD_WIDTH[default_variant] == 256:
                # A contiguous ``cos_sin`` view that starts 16 bytes into its allocation: the wide-load build
                # refuses it on every rank before any collective, naming the opt-out.
                buffer = torch.empty(
                    4 + batch * s_local * HEAD_DIM, dtype=torch.float32, device=device
                )
                misaligned = buffer[4:].view(batch * s_local, HEAD_DIM)
                misaligned.copy_(cos_sin)
                assert misaligned.is_contiguous() and misaligned.data_ptr() % 32 == 16
                with pytest.raises(
                    ValueError, match="FLASHINFER_QKV_EXCHANGE_PRODUCER_VARIANT=shipped"
                ):
                    default_session(x, w_qkv, q_scale, k_scale, misaligned)
                del buffer, misaligned
        finally:
            default_session.close()
            opt_out_session.close()
        torch.cuda.synchronize(device)
        dist.barrier(group=group)

        # 9. Cross-destination-band plan (the default of its (architecture, world size) cell where its build is
        #    delivered: the 256-column pair tile across band edges, one persistent launch) against the band plan
        #    (``xband_min_m=0``): bitwise-equal outputs; the bound programs differ by the three cross-destination-band
        #    stages only.  Every row of this group is eligible with 8 ranks (one head per destination).
        batch, s_local = 1, 4096
        xband_default = plans.default_xband_min_m(arch, world_size=world_size)
        xband_geometry = plans.Geometry(
            world_size=world_size, batch=batch, s_local=s_local, hidden=hidden
        )
        if (
            xband_default
            and batch * s_local >= xband_default
            and plans.xband_eligible(xband_geometry)
            and loader.has_xband_producer(arch, default_variant, fold=fold_default)
        ):
            x, w_qkv, q_scale, k_scale, cos_sin = _operands(
                device, batch=batch, s_local=s_local, hidden=hidden, seed=51 + rank
            )
            expected = _expected(
                x, w_qkv, q_scale, k_scale, cos_sin, group, world_size, rank
            )
            xband_path = _session_with({})
            band_path = _session_with({backend.SHAPE_GATES_ENV: "xband_min_m=0"})
            try:
                xband_info = xband_path.describe()
                band_info = band_path.describe()
                assert xband_info["xband_min_m"]["value"] == xband_default
                assert xband_info["xband_min_m"]["source"] == "default"
                assert xband_info["xband_min_m"]["eligible"] is True
                assert band_info["xband_min_m"]["value"] == 0
                assert band_info["xband_min_m"]["source"] == "env"
                assert band_info["shape_gates"]["source"] == "default"
                bound_xband = xband_path.bound_programs(batch, s_local)
                bound_band = band_path.bound_programs(batch, s_local)
                xb_build = loader.xband_variant(default_variant)
                assert bound_xband["producer_xband"] == loader.producer_program(
                    arch, plans.XBAND_BN, True, xb_build
                )
                for phase in loader.FOLD_PHASES if fold_default else ():
                    assert bound_xband[
                        f"producer_xband_fold{phase}"
                    ] == loader.fold_producer_program(
                        arch, plans.XBAND_BN, True, phase, xb_build
                    )
                assert {
                    key: name
                    for key, name in bound_xband.items()
                    if not key.startswith("producer_xband")
                } == bound_band
                xband_row = xband_path.describe()["prepared_row_plans"][
                    f"{batch}x{s_local}"
                ]
                band_row = band_path.describe()["prepared_row_plans"][
                    f"{batch}x{s_local}"
                ]
                assert xband_row["xband"] is True
                assert xband_row["block_n"] == plans.XBAND_BN
                assert "xband_min_m" in xband_row["fired"]
                assert xband_row["producer_launches"] == "persistent"
                assert xband_row["producer_parts"] == 1
                assert "xband" not in band_row
                assert "xband_min_m" not in band_row["fired"]
                assert xband_row["num_words"] == band_row["num_words"]
                for _round in range(3):
                    from_xband = tuple(
                        t.clone()
                        for t in xband_path(x, w_qkv, q_scale, k_scale, cos_sin)
                    )
                    from_band = band_path(x, w_qkv, q_scale, k_scale, cos_sin)
                    torch.cuda.synchronize(device)
                    _check_outputs(xband_path, from_xband, expected)
                    for one, other in zip(from_xband, from_band, strict=True):
                        assert torch.equal(one, other)
            finally:
                xband_path.close()
                band_path.close()
            torch.cuda.synchronize(device)
            dist.barrier(group=group)

        # 10. 128-column tile table (``HOPPER_BN128_MAX_M_BY_ARCH_WORLD``: SM90 with 4 ranks at hidden 3072): one
        #     256-column session runs a row at the bound on the 128-column producer and a larger row on its own; both
        #     against the reference, the bound programs and the prepared rows name the row's tile width.
        bn128_hidden = 3072
        bn128_bound = plans.hopper_bn128_max_m(arch, world_size, bn128_hidden)
        if bn128_bound:
            bn128_session = QkvNormRopeExchangeOverlap(
                group,
                hidden=bn128_hidden,
                num_heads=bn128_hidden // HEAD_DIM,
                layout="bshd",
                max_tokens=2 * bn128_bound,
                backend="nvlink",
                device=device,
            )
            try:
                info = bn128_session.describe()
                assert info["block_n"] == 256
                assert info["bn128_max_m"] == {
                    "value": bn128_bound,
                    "source": "default",
                    "default": bn128_bound,
                }
                launched = {
                    name
                    for bn in (plans.HOPPER_BN128_BN, 256)
                    for name in (
                        [
                            loader.fold_producer_program(
                                arch, bn, False, phase, default_variant
                            )
                            for phase in loader.FOLD_PHASES
                        ]
                        if fold_default
                        else [loader.producer_program(arch, bn, False, default_variant)]
                    )
                }
                assert launched <= set(info["programs"]["producers"])
                for s_local, row_bn in (
                    (bn128_bound, plans.HOPPER_BN128_BN),
                    (2 * bn128_bound, 256),
                ):
                    x, w_qkv, q_scale, k_scale, cos_sin = _operands(
                        device,
                        batch=1,
                        s_local=s_local,
                        hidden=bn128_hidden,
                        seed=61 + rank,
                    )
                    expected = _expected(
                        x, w_qkv, q_scale, k_scale, cos_sin, group, world_size, rank
                    )
                    bound = bn128_session.bound_programs(1, s_local)
                    assert bound["producer"] == loader.producer_program(
                        arch, row_bn, False, default_variant
                    )
                    row = bn128_session.describe()["prepared_row_plans"][f"1x{s_local}"]
                    assert row.get("block_n", 256) == row_bn
                    assert "xband" not in row
                    for _call in range(2):
                        _check_outputs(
                            bn128_session,
                            bn128_session(x, w_qkv, q_scale, k_scale, cos_sin),
                            expected,
                        )
                    torch.cuda.synchronize(device)
            finally:
                bn128_session.close()
            torch.cuda.synchronize(device)
            dist.barrier(group=group)
    finally:
        dist.destroy_process_group()


def _spawn(world_size, target=_run_worker):
    mp.set_start_method("spawn", force=True)
    port = random.randint(30000, 60000)
    processes = [
        mp.Process(
            target=target,
            args=(rank, world_size, port),
            name=f"qkv-exchange-{rank}",
        )
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=WORKER_TIMEOUT_SECONDS)
    alive = [process.name for process in processes if process.is_alive()]
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join()
    assert not alive, f"workers timed out after {WORKER_TIMEOUT_SECONDS} s: {alive}"
    failed = [
        (process.name, process.exitcode)
        for process in processes
        if process.exitcode != 0
    ]
    assert not failed, f"workers failed: {failed}"


def _supported_device():
    if not torch.cuda.is_available():
        return False
    return tuple(torch.cuda.get_device_capability(0)) in loader.CAPABILITY_ARCH


ODD_BAND_HIDDEN = 768


def _run_odd_band_worker(rank, world_size, port):
    """The default session of a three-head destination band (hidden 768, 6 heads over 2 ranks: ``block_n=128``):
    it constructs with the default build where that build is delivered for the tile width and with the baseline
    build otherwise (the two-rank ``fma`` defaults are delivered at ``block_n=256`` only), binds exactly the programs
    of the ``shipped`` opt-out in the second case, and matches the reference on a single-CTA row and a pair row; a
    ``cos_sin`` view that does not start at a 16-byte-aligned address is refused on every rank."""

    from flashinfer.comm import QkvNormRopeExchangeOverlap

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    group = dist.group.WORLD
    hidden = ODD_BAND_HIDDEN
    heads = hidden // HEAD_DIM
    arch = loader.device_facts(rank).arch
    max_tokens = plans.PAIR_MIN_M
    try:
        capacity = plans.Geometry(
            world_size=world_size, batch=1, s_local=max_tokens, hidden=hidden
        )
        assert plans.block_n_for(capacity) == 128
        default = loader.default_producer_variant(arch, world_size)
        delivered = any(
            loader.has_producer(arch, 128, pair, default) for pair in (False, True)
        )
        variant = default if delivered else loader.SHIPPED_VARIANT

        def session_with(env):
            previous = {name: os.environ.pop(name, None) for name in env}
            os.environ.update(env)
            try:
                return QkvNormRopeExchangeOverlap(
                    group,
                    hidden=hidden,
                    num_heads=heads,
                    layout="bshd",
                    max_tokens=max_tokens,
                    backend="nvlink",
                    device=device,
                )
            finally:
                for name, value in previous.items():
                    os.environ.pop(name, None)
                    if value is not None:
                        os.environ[name] = value

        os.environ.pop(backend.PRODUCER_VARIANT_ENV, None)
        session = session_with({})
        opt_out = session_with({backend.PRODUCER_VARIANT_ENV: "shipped"})
        try:
            info = session.describe()
            assert info["block_n"] == 128
            assert info["producer_variant"] == {
                "value": variant,
                "source": "default",
                "env": backend.PRODUCER_VARIANT_ENV,
                "default": default,
            }
            for batch, s_local, seed in ((1, 200, 61), (1, max_tokens, 71)):
                x, w_qkv, q_scale, k_scale, cos_sin = _operands(
                    device,
                    batch=batch,
                    s_local=s_local,
                    hidden=hidden,
                    seed=seed + rank,
                )
                expected = _expected(
                    x, w_qkv, q_scale, k_scale, cos_sin, group, world_size, rank
                )
                if variant == loader.SHIPPED_VARIANT:
                    bound = session.bound_programs(batch, s_local)
                    assert bound == opt_out.bound_programs(batch, s_local)
                for _call in range(2):
                    _check_outputs(
                        session, session(x, w_qkv, q_scale, k_scale, cos_sin), expected
                    )
                    _check_outputs(
                        opt_out, opt_out(x, w_qkv, q_scale, k_scale, cos_sin), expected
                    )
            # A contiguous ``cos_sin`` view that starts 4 bytes into its allocation: refused before any collective.
            buffer = torch.empty(
                1 + batch * s_local * HEAD_DIM, dtype=torch.float32, device=device
            )
            misaligned = buffer[1:].view(batch * s_local, HEAD_DIM)
            misaligned.copy_(cos_sin)
            assert misaligned.data_ptr() % 16 == 4
            for target in (session, opt_out):
                with pytest.raises(ValueError, match="cos_sin must start at a 16-byte"):
                    target(x, w_qkv, q_scale, k_scale, misaligned)
            del buffer, misaligned
        finally:
            session.close()
            opt_out.close()
        torch.cuda.synchronize(device)
        dist.barrier(group=group)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_qkv_norm_rope_exchange_overlap_multi_gpu(world_size):
    if not loader.ROUTES:
        pytest.skip("program tables are filled by the exporter")
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} GPUs")
    if not _supported_device():
        pytest.skip("requires SM90, SM100, SM103 or SM107 devices")
    arch = loader.device_facts(0).arch
    if world_size not in loader.supported_world_sizes(arch):
        pytest.skip(f"world size {world_size} is not supported on {arch}")
    _spawn(world_size)


def test_qkv_norm_rope_exchange_overlap_multi_gpu_default_odd_band():
    """Two ranks, hidden 768 (6 heads, a three-head destination band, ``block_n=128``): the default session
    constructs and runs on every architecture that delivers the 128-column tile (:func:`_run_odd_band_worker`)."""

    if not loader.ROUTES:
        pytest.skip("program tables are filled by the exporter")
    if torch.cuda.device_count() < 2:
        pytest.skip("requires 2 GPUs")
    if not _supported_device():
        pytest.skip("requires SM90, SM100, SM103 or SM107 devices")
    arch = loader.device_facts(0).arch
    if 2 not in loader.supported_world_sizes(arch):
        pytest.skip(f"world size 2 is not supported on {arch}")
    if not any(loader.has_producer(arch, 128, pair) for pair in (False, True)):
        pytest.skip(f"{arch} delivers no block_n=128 producer")
    _spawn(2, _run_odd_band_worker)


def test_math_reference_is_self_consistent():
    """The fp32 references used by the multi-GPU tests: unit scales and zero angles leave a normalised head."""

    x = torch.randn(1, 4, 256, dtype=torch.bfloat16)
    w = torch.eye(256, dtype=torch.bfloat16).repeat(3, 1)
    ones = torch.ones(HEAD_DIM, dtype=torch.bfloat16)
    cos_sin = torch.cat(
        [torch.ones(4, HEAD_DIM // 2), torch.zeros(4, HEAD_DIM // 2)], dim=-1
    )
    for reference in (_reference_local, _reference_per_operator):
        q, _k, v = reference(x, w, ones, ones, cos_sin)
        assert torch.allclose(v.float(), x.view(1, 4, 2, HEAD_DIM).float())
        rms = q.float().pow(2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=2e-2)
        assert math.isfinite(float(q.float().abs().max()))


def test_per_operator_reference_mirrors_its_rounding_points():
    """``_reference_per_operator`` differs from ``_reference_local`` only by the rounding points it mirrors: the same
    stored ``v``; q / k not bit-identical and within 0.1 of each other (two BF16 rounding chains -- the benchmark
    guard's tolerance; the recorded differences reach 0.0625 at larger shapes, which is why the per-operator path is
    compared with the reference of its own chain).  ``_check`` admits a difference inside its tolerance and rejects
    one just beyond it."""

    x, w_qkv, q_scale, k_scale, cos_sin = _operands(
        torch.device("cpu"), batch=1, s_local=256, hidden=512, seed=41
    )
    fused = _reference_local(x, w_qkv, q_scale, k_scale, cos_sin)
    per_operator = _reference_per_operator(x, w_qkv, q_scale, k_scale, cos_sin)
    assert torch.equal(fused[2], per_operator[2])
    for index in range(2):
        assert not torch.equal(per_operator[index], fused[index])
        torch.testing.assert_close(
            per_operator[index].float(), fused[index].float(), atol=0.1, rtol=1e-2
        )
    q = per_operator[0]
    _check(q, q)
    _check(q.float() + 0.049, q)
    with pytest.raises(AssertionError, match="Greatest absolute difference"):
        _check(q.float() + 0.0625, q)
