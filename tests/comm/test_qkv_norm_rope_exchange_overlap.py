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
request validation and the loader tables.  The multi-GPU tests (two, four or
eight devices of one supported architecture) check both layouts against an
fp32 step-by-step reference including a tail row count, CUDA-graph capture and
replay with fresh inputs and eager calls in between, two sessions in flight on
one group, the refusals that protect a captured graph, and the NCCL path.
"""

from __future__ import annotations

import math
import random

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from flashinfer.comm import cake_qkv_exchange_plan as plans
from flashinfer.comm.qkv_norm_rope_exchange_overlap import validate_request
from flashinfer.jit import cake_qkv_norm_rope_exchange_overlap as loader

HEAD_DIM = 128
EPS = 1e-6
WORKER_TIMEOUT_SECONDS = 1800


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
                assert record["cluster"] == (
                    [2, 1, 1] if key.endswith("_pair") else [1, 1, 1]
                )
                assert set(record["args"]) >= {"A", "B", "tiles", "ready_target"}
            elif key.startswith("barrier_p"):
                assert record["args"][0] == "device_index"
            else:
                assert record["args"][0] == "ready"
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


def _redistribute(local, group, world_size, rank):
    """``[B, S_local, H, D] -> [B, S_global, H_local, D]``: this rank's head band of every rank's tokens."""

    peers = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(peers, local.contiguous(), group=group)
    h_local = local.shape[2] // world_size
    return torch.cat(
        [peer[:, :, rank * h_local : (rank + 1) * h_local, :] for peer in peers], dim=1
    )


def _expected(x, w_qkv, q_scale, k_scale, cos_sin, group, world_size, rank):
    q, k, v = _reference_local(x, w_qkv, q_scale, k_scale, cos_sin)
    return tuple(_redistribute(t, group, world_size, rank) for t in (q, k, v))


def _check(actual, expected):
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected.float(), atol=1e-2, rtol=1e-2)


def _check_outputs(session, outputs, expected):
    if session.layout == "bshd":
        for actual, want in zip(outputs, expected, strict=True):
            _check(actual, want)
    else:
        for index, want in enumerate(expected):
            _check(outputs[..., index * HEAD_DIM : (index + 1) * HEAD_DIM], want)


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
            eager_x = torch.randn_like(x)
            for _replay in range(3):
                x.copy_(torch.randn_like(x))
                dist.barrier(group=group)
                graph.replay()
                torch.cuda.synchronize(device)
                replay_expected = _expected(
                    x, w_qkv, q_scale, k_scale, cos_sin, group, world_size, rank
                )
                _check_outputs(session, first, replay_expected)
                _check_outputs(session, second, replay_expected)
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

        # 4. NCCL path: same math through the per-operator path; not capturable.
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
            first = nccl(x, w_qkv, q_scale, k_scale, cos_sin)
            _check_outputs(nccl, first, expected)
            # A call's outputs stay valid for num_slots - 1 further calls on this path too:
            # the next call lands in another receive slot and leaves the first views intact.
            kept = tuple(t.clone() for t in first)
            nccl(x.neg(), w_qkv, q_scale, k_scale, cos_sin)
            torch.cuda.synchronize(device)
            for before, view in zip(kept, first, strict=True):
                assert torch.equal(before, view)
            _check_outputs(nccl, nccl(x, w_qkv, q_scale, k_scale, cos_sin), expected)
            refusal = torch.cuda.CUDAGraph()
            with (
                pytest.raises(RuntimeError, match="not CUDA-graph capturable"),
                torch.cuda.graph(refusal, stream=torch.cuda.Stream(device=device)),
            ):
                nccl(x, w_qkv, q_scale, k_scale, cos_sin)
            del refusal
        finally:
            nccl.close()

        # 5. Heads not divisible by the world size raise on every rank before any collective.
        with pytest.raises(ValueError, match="divisible by world_size"):
            QkvNormRopeExchangeOverlap(
                group,
                hidden=3584,
                num_heads=28,
                layout="bshd",
                max_tokens=128,
                backend="nvlink",
                device=device,
            )
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
    """The fp32 reference used by the multi-GPU tests: unit scales and zero angles leave a normalised head."""

    x = torch.randn(1, 4, 256, dtype=torch.bfloat16)
    w = torch.eye(256, dtype=torch.bfloat16).repeat(3, 1)
    ones = torch.ones(HEAD_DIM, dtype=torch.bfloat16)
    cos_sin = torch.cat(
        [torch.ones(4, HEAD_DIM // 2), torch.zeros(4, HEAD_DIM // 2)], dim=-1
    )
    q, _k, v = _reference_local(x, w, ones, ones, cos_sin)
    assert torch.allclose(v.float(), x.view(1, 4, 2, HEAD_DIM).float())
    rms = q.float().pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=2e-2)
    assert math.isfinite(float(q.float().abs().max()))
