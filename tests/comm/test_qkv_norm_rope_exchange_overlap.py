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
three), the large-row gates against the persistent launch, and the baseline
producer build and group scaling (the two environment opt-outs) against the
defaults.  Every comparison of a device output with a host reference is
``_check``: finite, then ``assert_close`` at ``atol = 0.05``, ``rtol = 1e-2``.
Six comparisons are bitwise (``torch.equal``): the replayed outputs of the two
captured calls against each other (part 2), the kept slot views of the NCCL
path (part 4), the folded against the barrier launch path (part 5), the gated
against the persistent launch (part 7), the opt-outs against the defaults
(part 8) and, host-only, the ``v`` of the two references.
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
    calls of at least 16384 rows, and the per-destination launch gate off; on SM107 every gate off -- plus the one
    (architecture, world size) cell that differs from it: the small-row word gate on every row for 8 ranks on SM90."""

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
    # The (architecture, world size) cells: exactly the one new (architecture, world size) default, differing from
    # its row.
    overrides = plans.SHAPE_GATE_DEFAULT_BY_ARCH_WORLD
    assert set(overrides) == {("sm_90a", 8)}
    assert overrides[("sm_90a", 8)] == {
        "words_capacity_small_m": plans.SHAPE_GATE_ALL_ROWS
    }
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
        # The small-row word gate on every row (``all``) in either direction: the SM90 8-rank default, an opt-in
        # elsewhere; the previous SM90 8-rank plan is the explicit threshold 4096.
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
        # the baseline plan on SM107 (not measured there)
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

    assert loader.PRODUCER_VARIANTS == ("", "ow256", "bulk_ow256", "bulk")
    assert loader.SHIPPED_VARIANT == ""
    assert loader.PRODUCER_VARIANT_OPERAND_LOAD_WIDTH == {
        "": 128,
        "ow256": 256,
        "bulk_ow256": 256,
        "bulk": 128,
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


def test_producer_variant_default_table():
    """The default producer build per architecture and world size (the table the exporter renders), and its
    delivery: every default build is delivered with both rendezvous phases for at least one tile width, every
    variant program has its baseline twin, and ``delivered_producer_variants`` reports exactly the route table."""

    if not loader.ROUTES:
        pytest.skip("program tables are filled by the exporter")
    table = loader.PRODUCER_VARIANT_DEFAULT_BY_ARCH_WORLD
    assert table == {
        "sm_90a": {2: "", 4: "", 8: ""},
        "sm_100a": {2: "", 4: "ow256", 8: "bulk_ow256"},
        "sm_103a": {2: "", 4: "ow256", 8: "ow256"},
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
                loader.parse_producer_key(key).variant == ""
                for key in routes
                if key.startswith("producer_bn")
            )


def test_producer_variant_resolution(monkeypatch):
    """Environment over the (architecture, world size) default: ``auto`` keeps the table, ``shipped`` selects the
    baseline build, a delivered name selects that build; an undelivered build fails closed naming the environment
    variable, the ``shipped`` opt-out and the delivered builds -- also when the default itself is not delivered."""

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
                for value in (None, "auto", ""):
                    if value is None:
                        monkeypatch.delenv(env, raising=False)
                    else:
                        monkeypatch.setenv(env, value)
                    if default in delivered:
                        resolved = resolve(arch, world_size, block_n)
                        assert resolved == (default, "default")
                    else:
                        with pytest.raises(ValueError, match=f"{env}=shipped"):
                            resolve(arch, world_size, block_n)
                monkeypatch.setenv(env, "shipped")
                assert resolve(arch, world_size, block_n) == ("", "env")
                for name in ("ow256", "bulk_ow256", "bulk"):
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
                    if parsed.variant:
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
        #    (the architecture default on SM100 / SM103, the explicit environment override elsewhere; on SM90 the
        #    default large-row reserved-SM gate fires beside it) against the persistent launch (every gate off)
        #    -- bitwise-equal outputs, one producer launch per destination run; on SM90 also against the explicit
        #    ``reserved_sms_large_m=0,words_capacity_small_m=0`` opt-out (the reserved-off session switches the word
        #    gate off too, so the 8-rank ``all`` default does not fire on this row and nothing fires on that session).
        batch, s_local = 1, 16384
        x, w_qkv, q_scale, k_scale, cos_sin = _operands(
            device, batch=batch, s_local=s_local, hidden=hidden, seed=31 + rank
        )
        expected = _expected(
            x, w_qkv, q_scale, k_scale, cos_sin, group, world_size, rank
        )
        default_gates = plans.default_shape_gates(arch, world_size=world_size)
        # The verdict is read with the row's tile width, as the backend reads it: the pair-tile gate of the
        # (architecture, world size) table fires on 256-column rows only.
        block_n = plans.block_n_for(
            plans.Geometry(
                world_size=world_size, batch=batch, s_local=s_local, hidden=hidden
            )
        )
        default_verdict = plans.gate_rows(default_gates, batch * s_local, bn=block_n)
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
            else "per_dest_large_m=16384,words_capacity_small_m=0"
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
            geometry = plans.Geometry(
                world_size=world_size, batch=batch, s_local=s_local, hidden=hidden
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
    finally:
        dist.destroy_process_group()


def _spawn(world_size):
    mp.set_start_method("spawn", force=True)
    port = random.randint(30000, 60000)
    processes = [
        mp.Process(
            target=_run_worker,
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
