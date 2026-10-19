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
gates (``shape_gates``) and the
loader tables.  The multi-GPU tests (two, four or eight devices of one
supported architecture) check both layouts against an fp32 step-by-step
reference including a tail row count, CUDA-graph capture and replay with fresh
inputs and eager calls in between, two sessions in flight on one group, the
refusals that protect a captured graph, the NCCL path (the per-operator path:
cuBLAS projection, ``rmsnorm``, RoPE) against a reference that mirrors its own
rounding points (the ranks agree on the tolerance verdict before raising), and
the two launch paths against each other (the default folded path two launches
per call, the explicit barrier path three).  Every comparison of a device
output with a host reference is ``_check``: finite, then ``assert_close`` at
``atol = 0.05``, ``rtol = 1e-2``.  Four comparisons are bitwise
(``torch.equal``): the replayed outputs of the two captured calls against each
other (part 2), the kept slot views of the NCCL path (part 4), the folded
against the barrier launch path (part 5) and, host-only, the ``v`` of the two
references.
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
    for arch in ("sm_90a", "sm_100a"):
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
    large = plans.Geometry(world_size=2, batch=1, s_local=65536, hidden=8192)
    policy = plans.plan_policy(large, arch="sm_103a")
    assert policy.pair and policy.order == "grouped" and policy.group_m % 2 == 0
    assert policy.group_m <= policy.m_tiles // plans.PAIR_MIN_GROUPS
    hopper = plans.plan_policy(large, arch="sm_90a")
    assert hopper.order == "grouped" and not hopper.pair


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


def test_fold_barrier_default_table():
    """The default launch path per architecture: the folded rendezvous on SM90, SM100 and SM103 alike; the
    separate barrier kernel is the explicit ``fold_barrier=False`` path."""

    assert loader.FOLD_BARRIER_DEFAULT_BY_ARCH == {
        "sm_90a": True,
        "sm_100a": True,
        "sm_103a": True,
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
    """Keyword over environment over architecture default; every delivered architecture folds by default and
    delivers the folded producers; ``True`` fails closed on a route table without them."""

    if not loader.ROUTES:
        pytest.skip("program tables are filled by the exporter")
    env = backend.FOLD_BARRIER_ENV
    for arch in loader.ROUTES:
        assert loader.default_fold_barrier(arch) is True
        assert loader.fold_producers_delivered(arch)
        monkeypatch.delenv(env, raising=False)
        assert backend.resolve_fold_barrier(arch) == (True, "default")
        assert backend.resolve_fold_barrier(arch, False) == (False, "keyword")
        monkeypatch.setenv(env, "auto")
        assert backend.resolve_fold_barrier(arch) == (True, "default")
        monkeypatch.setenv(env, "0")
        assert backend.resolve_fold_barrier(arch) == (False, "env")
        monkeypatch.setenv(env, "1")
        assert backend.resolve_fold_barrier(arch) == (True, "env")
        monkeypatch.setenv(env, "0")
        assert backend.resolve_fold_barrier(arch, True) == (True, "keyword")
        monkeypatch.setenv(env, "yes")
        with pytest.raises(ValueError, match=env):
            backend.resolve_fold_barrier(arch)
        monkeypatch.delenv(env, raising=False)
        with pytest.raises(ValueError, match="bool or None"):
            backend.resolve_fold_barrier(arch, 1)  # type: ignore[arg-type]
    # A route table without folded producers (none delivered; a patched table): the default and the explicit
    # ``True`` fail closed naming the opt-out, ``False`` resolves.
    arch = next(iter(loader.ROUTES))
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
    """The row-count gates per architecture: one readiness word per destination for calls of at most 4096 rows and one
    producer launch per destination run for calls of at least 16384 rows on SM100 / SM103; both off on SM90."""

    assert plans.SHAPE_GATE_NAMES == ("words_capacity_small_m", "per_dest_large_m")
    assert plans.SHAPE_GATE_OFF == 0
    assert plans.SHAPE_GATE_DEFAULT_BY_ARCH == {
        "sm_90a": {"words_capacity_small_m": 0, "per_dest_large_m": 0},
        "sm_100a": {"words_capacity_small_m": 4096, "per_dest_large_m": 16384},
        "sm_103a": {"words_capacity_small_m": 4096, "per_dest_large_m": 16384},
    }
    assert set(plans.SHAPE_GATE_DEFAULT_BY_ARCH) == set(loader.ARCH_FLAGS)
    for arch, table in plans.SHAPE_GATE_DEFAULT_BY_ARCH.items():
        assert (
            plans.default_shape_gates(arch) == table
            and plans.default_shape_gates(arch) is not table
        )
        assert plans.check_shape_gates(table) == table
    with pytest.raises(ValueError, match="no shape gates"):
        plans.default_shape_gates("sm_120a")
    for bad in (
        {"words_capacity_small_m": 1},
        {"words_capacity_small_m": -1, "per_dest_large_m": 0},
        {"words_capacity_small_m": True, "per_dest_large_m": 0},
    ):
        with pytest.raises(ValueError):
            plans.check_shape_gates(bad)
    # Inclusive thresholds; a gate at 0 never fires.
    blackwell = plans.default_shape_gates("sm_100a")
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
    hopper = plans.default_shape_gates("sm_90a")
    for rows in (128, 4096, 16384, 65536):
        verdict = plans.gate_rows(hopper, rows)
        assert verdict.fired == () and verdict.producer_launches == "persistent"
        assert (
            verdict.words_capacity_per_dest is None
            and verdict.summary()["rows"] == rows
        )
    with pytest.raises(ValueError, match="positive int"):
        plans.gate_rows(blackwell, 0)


def test_shape_gates_resolution(monkeypatch):
    """Environment over architecture default: ``auto`` keeps the table, ``off`` switches every gate off, a partial
    ``gate=value`` list overrides only the named gates; malformed values are errors."""

    env = backend.SHAPE_GATES_ENV
    for arch in plans.SHAPE_GATE_DEFAULT_BY_ARCH:
        monkeypatch.delenv(env, raising=False)
        assert backend.resolve_shape_gates(arch) == (
            plans.default_shape_gates(arch),
            "default",
        )
        monkeypatch.setenv(env, "auto")
        assert backend.resolve_shape_gates(arch) == (
            plans.default_shape_gates(arch),
            "default",
        )
        monkeypatch.setenv(env, "off")
        assert backend.resolve_shape_gates(arch) == (
            {"words_capacity_small_m": 0, "per_dest_large_m": 0},
            "env",
        )
        monkeypatch.setenv(env, "per_dest_large_m=0")
        gates, source = backend.resolve_shape_gates(arch)
        assert source == "env" and gates["per_dest_large_m"] == 0
        assert (
            gates["words_capacity_small_m"]
            == plans.default_shape_gates(arch)["words_capacity_small_m"]
        )
        monkeypatch.setenv(env, "words_capacity_small_m=2048; per_dest_large_m=auto")
        gates, source = backend.resolve_shape_gates(arch)
        assert source == "env" and gates == {
            "words_capacity_small_m": 2048,
            "per_dest_large_m": plans.default_shape_gates(arch)["per_dest_large_m"],
        }
        for bad in (
            "yes",
            "words_capacity_small_m",
            "reserved_sms_large_m=1",
            "per_dest_large_m=-3",
            "per_dest_large_m=x",
        ):
            monkeypatch.setenv(env, bad)
            with pytest.raises(ValueError, match=env):
                backend.resolve_shape_gates(arch)
    monkeypatch.delenv(env, raising=False)
    assert backend.shape_gates_from_env() is None


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
        # The verdict of this row on each architecture.
        verdict = plans.gate_rows(plans.default_shape_gates(arch), geometry.rows)
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
                assert record["cluster"] == ([2, 1, 1] if "_pair" in key else [1, 1, 1])
                assert set(record["args"]) >= {"A", "B", "tiles", "ready_target"}
                if "_fold" in key:
                    # The folded producer: the plain producer's ABI plus the rendezvous process-group
                    # parameters before the grid; the rendezvous warp is one extra warp per CTA on
                    # SM100 / SM103 and the otherwise idle TMA warp on SM90 (same thread count).
                    plain = loader.PROGRAMS[table[key.rsplit("_fold", 1)[0]]]
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
            elif key.startswith("barrier_p"):
                assert record["args"][0] == "device_index"
            else:
                assert record["args"][0] == "ready"
        # Folded producers: both rendezvous phases of every routed producer on every architecture (the default
        # launch path folds the rendezvous everywhere).
        plain_keys = {
            key for key in table if key.startswith("producer_bn") and "_fold" not in key
        }
        complete = {
            f"{key}_fold{phase}" for key in plain_keys for phase in loader.FOLD_PHASES
        }
        fold_keys = {key for key in table if "_fold" in key}
        assert loader.default_fold_barrier(arch) and fold_keys == complete
        assert loader.fold_producers_delivered(arch)
        # The policy never selects a producer the architecture does not deliver.
        for hidden, world_size, rows in (
            (3072, 8, 512),
            (3072, 8, 8192),
            (4096, 8, 512),
            (8192, 2, 65536),
        ):
            geometry = plans.Geometry(
                world_size=world_size, batch=1, s_local=rows, hidden=hidden
            )
            policy = plans.plan_policy(
                geometry,
                arch=arch,
                pair_available=loader.has_producer(
                    arch, plans.block_n_for(geometry), True
                ),
            )
            assert loader.has_producer(arch, policy.bn, policy.pair)
            assert loader.has_fold_producer(
                arch, policy.bn, policy.pair
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

        # 5. Launch paths: the architecture default (the folded rendezvous on every architecture) against the
        #    explicit barrier path -- bitwise-equal outputs, two versus three launches per call.
        arch = loader.device_facts(device.index).arch
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
            assert (
                default_info["fold_barrier"]
                is loader.default_fold_barrier(arch)
                is True
            )
            assert default_info["fold_barrier_requested"] == "auto"
            assert default_info["launches_per_call"] == 2
            assert default_info["launch_sequence"] == ["producer_fold<phase>", "wait"]
            assert barrier_info["fold_barrier"] is False
            assert barrier_info["fold_barrier_requested"] is False
            assert barrier_info["launches_per_call"] == 3
            bound = default_path.bound_programs(batch, s_local)
            assert bound == barrier_path.bound_programs(batch, s_local)
            assert "producer_fold0" in bound and "producer_fold1" in bound
            assert "barrier_p0" in bound and "producer" in bound and len(bound) == 6
            # The row-count gates of the architecture default on this row (``batch * s_local`` = 512 rows): the
            # resolved table, its source and the prepared row's verdict agree with the plan module.
            gates = default_path.describe()["shape_gates"]
            assert (
                gates["values"] == plans.default_shape_gates(arch) == gates["default"]
            )
            assert (
                gates["source"] == "default" and gates["env"] == backend.SHAPE_GATES_ENV
            )
            verdict = plans.gate_rows(gates["values"], batch * s_local)
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
            assert (
                row_plan["producer_parts"] == 1 and row_plan["launches_per_call"] == 2
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
        #    (the architecture default on SM100 / SM103, the explicit environment override elsewhere) against the
        #    persistent launch (every gate off) -- bitwise-equal outputs, one producer launch per destination run.
        batch, s_local = 1, 16384
        x, w_qkv, q_scale, k_scale, cos_sin = _operands(
            device, batch=batch, s_local=s_local, hidden=hidden, seed=31 + rank
        )
        expected = _expected(
            x, w_qkv, q_scale, k_scale, cos_sin, group, world_size, rank
        )
        default_gates = plans.default_shape_gates(arch)
        default_is_per_dest = (
            plans.gate_rows(default_gates, batch * s_local).producer_launches
            == "per_dest"
        )
        gated_env = (
            None
            if default_is_per_dest
            else "per_dest_large_m=16384,words_capacity_small_m=0"
        )

        def _session_under(env_value):
            previous = os.environ.pop(backend.SHAPE_GATES_ENV, None)
            if env_value is not None:
                os.environ[backend.SHAPE_GATES_ENV] = env_value
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
                os.environ.pop(backend.SHAPE_GATES_ENV, None)
                if previous is not None:
                    os.environ[backend.SHAPE_GATES_ENV] = previous

        per_dest_path = _session_under(gated_env)
        persistent_path = _session_under("off")
        try:
            gated_info = per_dest_path.describe()
            plain_info = persistent_path.describe()
            assert gated_info["shape_gates"]["source"] == (
                "default" if default_is_per_dest else "env"
            )
            assert plain_info["shape_gates"]["source"] == "env"
            assert plain_info["shape_gates"]["values"] == {
                "words_capacity_small_m": 0,
                "per_dest_large_m": 0,
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
            assert (
                plain_row["producer_launches"] == "persistent"
                and plain_row["fired"] == []
            )
            geometry = plans.Geometry(
                world_size=world_size, batch=batch, s_local=s_local, hidden=hidden
            )
            block_n = gated_info["block_n"]
            _policy, plan = plans.plan_for(
                geometry,
                "bshd",
                rank,
                arch=arch,
                pair_available=gated_info["pair_available"][block_n],
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
        finally:
            per_dest_path.close()
            persistent_path.close()
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
        pytest.skip("requires SM90, SM100 or SM103 devices")
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
