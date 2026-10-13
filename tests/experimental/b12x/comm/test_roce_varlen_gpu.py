"""Correctness tests for RoCEnante's variable-length all-gather.

Run with torchrun on two or more nodes that share a RoCE fabric, for example
from every node of a DGX Spark cluster::

    NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 NCCL_IB_GID_INDEX=3 \\
    torchrun --nnodes=2 --nproc-per-node=1 --node-rank=$RANK \\
        --master-addr=$MASTER --master-port=29650 \\
        -m pytest -x tests/experimental/b12x/comm/test_roce_varlen_gpu.py

Without a torchrun environment the tests skip.
"""

from __future__ import annotations

import os
import time
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from b12x.preparation import PreparationSession, PreparedCall

pytestmark = pytest.mark.skipif(
    "WORLD_SIZE" not in os.environ or int(os.environ.get("WORLD_SIZE", "1")) < 2,
    reason="requires a torchrun launch with WORLD_SIZE >= 2",
)

CAPACITY = 4 << 20
# Small chunks so modest contributions span many chunk flags.
CHUNK = 64 << 10
SENTINEL = 0x5A


class _Prepared:
    """A runtime plus its published plan."""

    def __init__(self, runtime, plan):
        self.runtime, self.plan = runtime, plan

    def gather(self, source, counts, **kwargs):
        return self.runtime.all_gather_varlen(source, counts, plan=self.plan, **kwargs)

    def all_reduce(self, inp, **kwargs):
        return self.runtime.all_reduce(inp, plan=self.plan, **kwargs)


def _prepare(spin_limit: int | None = None, **kwargs) -> _Prepared:
    """A prepared runtime with a variable-length area over the torchrun world."""
    from b12x.comm import roce
    from b12x.comm.roce import _preparation

    if not roce.is_supported():
        pytest.skip("RoCE needs an integrated GPU with an active RDMA device")
    if not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(seconds=60))
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    previous = os.environ.get("B12X_ROCE_SPIN_LIMIT")
    if spin_limit is not None:
        os.environ["B12X_ROCE_SPIN_LIMIT"] = str(spin_limit)
    try:
        rt = roce.AllReduce(
            exchange_group=dist.group.WORLD,
            device=device,
            max_size=1 << 20,
            max_gather_bytes=1 << 20,
            pipe_bytes=8 << 20,
            max_varlen_bytes=CAPACITY,
            varlen_chunk_bytes=CHUNK,
            **kwargs,
        )
    finally:
        if spin_limit is not None:
            if previous is None:
                del os.environ["B12X_ROCE_SPIN_LIMIT"]
            else:
                os.environ["B12X_ROCE_SPIN_LIMIT"] = previous
    query = roce.query_from_runtime(
        rt,
        surface="AllReduce.all_gather_varlen",
        call={"dtypes": ("bfloat16",)},
        topology="roce_rdma",
        peer_hosts=tuple(f"rank-{r}" for r in range(rt.world_size)),
    )
    plan = roce.plan(query, runtime=rt)
    source = torch.zeros(4096, dtype=torch.uint8, device=device)
    counts = torch.full((rt.world_size,), 16, dtype=torch.int64, device=device)
    seed = torch.zeros(8, dtype=torch.bfloat16, device=device)

    def prepare(state):
        calls = (
            _preparation.prepared_varlen_call(state, source=source, counts=counts),
            _preparation.prepared_call(state, inp=seed),
        )
        return PreparedCall(run=lambda: [call.run() for call in calls])

    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare((plan.request(name="roce", prepare_call=prepare),))
    return _Prepared(rt, plan)


@pytest.fixture(scope="module")
def runtime():
    prepared = _prepare()
    yield prepared
    prepared.runtime.close()
    dist.barrier()


def _contribution(rank: int, nbytes: int, trial: int, device) -> torch.Tensor:
    """Rank-specific bytes every rank can regenerate."""
    gen = torch.Generator(device=device).manual_seed(1009 * trial + rank)
    return torch.randint(0, 256, (CAPACITY,), dtype=torch.uint8, device=device, generator=gen)


def _check(out: torch.Tensor, counts: list[int], trial: int) -> None:
    device = out.device
    rows = out.view(len(counts), -1)
    for peer, count in enumerate(counts):
        expected = _contribution(peer, count, trial, device)[:count]
        assert torch.equal(rows[peer, :count], expected), f"row {peer} differs"
        assert bool((rows[peer, count:] == SENTINEL).all()), f"row {peer} written past its count"


def _counts_cases(world: int) -> list[list[int]]:
    return [
        [16 * 1000] * world,
        [0] + [CHUNK * 3 + 16] * (world - 1),  # one rank sends nothing
        [16 * (1 + 4097 * r) for r in range(world)],  # differ by rank
        [CAPACITY] * world,
        [CAPACITY if r == world - 1 else 0 for r in range(world)],
        [0] * world,
    ]


def test_gathers_counts_that_differ_by_rank_and_call(runtime):
    """Every row holds its rank's first count bytes; nothing past a count is written."""
    world, rank = dist.get_world_size(), dist.get_rank()
    device = runtime.runtime.device
    for trial, counts in enumerate(_counts_cases(world)):
        source = _contribution(rank, counts[rank], trial, device)
        out = torch.full((world, CAPACITY), SENTINEL, dtype=torch.uint8, device=device)
        counts_tensor = torch.tensor(counts, dtype=torch.int64, device=device)
        runtime.gather(source, counts_tensor, out=out)
        torch.cuda.synchronize()
        _check(out, counts, trial)
    runtime.runtime.check_health()


def test_source_may_be_this_ranks_output_row(runtime):
    """Packing straight into our row of the output skips the local copy and still gathers."""
    world, rank = dist.get_world_size(), dist.get_rank()
    device = runtime.runtime.device
    counts = [16 * (3000 + 7 * r) for r in range(world)]
    out = torch.full((world, CAPACITY), SENTINEL, dtype=torch.uint8, device=device)
    out[rank, : counts[rank]] = _contribution(rank, counts[rank], 41, device)[: counts[rank]]
    runtime.gather(out[rank], torch.tensor(counts, dtype=torch.int64, device=device), out=out)
    torch.cuda.synchronize()
    _check(out, counts, 41)


def test_graph_replays_with_counts_changing_between_replays(runtime):
    """One captured graph mixes the gather with one-shot and two-shot reductions; counts change per replay."""
    world, rank = dist.get_world_size(), dist.get_rank()
    rt = runtime.runtime
    device = rt.device
    counts_tensor = torch.zeros(world, dtype=torch.int64, device=device)
    source = torch.empty(CAPACITY, dtype=torch.uint8, device=device)
    out = torch.empty((world, CAPACITY), dtype=torch.uint8, device=device)
    small = torch.randn(4096, dtype=torch.bfloat16, device=device)
    large = torch.randn(3 << 19, dtype=torch.bfloat16, device=device)
    small_out, large_out = torch.empty_like(small), torch.empty_like(large)
    expected_small, expected_large = small.clone(), large.clone()
    dist.all_reduce(expected_small)
    dist.all_reduce(expected_large)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        runtime.gather(source, counts_tensor, out=out)
        runtime.all_reduce(small, out=small_out)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream), rt.capture(stream=stream):
        runtime.all_reduce(small, out=small_out)
        runtime.gather(source, counts_tensor, out=out)
        runtime.all_reduce(large, out=large_out)
        runtime.gather(source, counts_tensor, out=out)
    for trial, counts in enumerate(_counts_cases(world) * 2):
        source.copy_(_contribution(rank, counts[rank], 100 + trial, device))
        counts_tensor.copy_(torch.tensor(counts, dtype=torch.int64))
        out.fill_(SENTINEL)
        small_out.zero_()
        large_out.zero_()
        torch.cuda.synchronize()
        dist.barrier()
        graph.replay()
        torch.cuda.synchronize()
        rt.check_health()
        _check(out, counts, 100 + trial)
        torch.testing.assert_close(small_out, expected_small, rtol=1e-2, atol=4e-2 * world)
        torch.testing.assert_close(large_out, expected_large, rtol=1e-2, atol=4e-2 * world)
    dist.barrier()


def test_proxy_catches_up_after_the_next_gather_overwrote_ready_words(runtime):
    """A proxy that sleeps across two gathers posts both: the second's ready words release the first.

    Rank 1's first gather completes on its peers' data alone, so its second
    stages every chunk (overwriting the ready words) before the stopped proxy
    posted any chunk of the first.
    """
    world, rank = dist.get_world_size(), dist.get_rank()
    rt = runtime.runtime
    device = rt.device
    counts = [CHUNK * 5 + 16 * (r + 1) for r in range(world)]
    counts_tensor = torch.tensor(counts, dtype=torch.int64, device=device)
    sources = [_contribution(rank, counts[rank], 200 + i, device) for i in range(2)]
    outs = [torch.full((world, CAPACITY), SENTINEL, dtype=torch.uint8, device=device) for _ in range(2)]
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 1:
        rt._proxy.stop()
    for source, out in zip(sources, outs, strict=True):
        runtime.gather(source, counts_tensor, out=out)
    if rank == 1:
        posted = rt._proxy.stats()["last_seq"]
        deadline = time.monotonic() + 20
        while int(rt._ctrl_words[0].item()) != posted + 2:
            assert time.monotonic() < deadline, "second gather never rang"
            time.sleep(0.001)
        rt._proxy.start()
    torch.cuda.synchronize()
    rt.check_health()
    for i, out in enumerate(outs):
        _check(out, counts, 200 + i)
    dist.barrier()


def test_unusable_count_stops_every_rank_before_sending():
    """A count that is not a multiple of 16 poisons every rank alike, with no timeout and no epoch."""
    world = dist.get_world_size()
    prepared = _prepare(spin_limit=2_000_000)
    rt = prepared.runtime
    try:
        epoch_before = rt.stats()["epoch"]
        counts = torch.tensor([16] * (world - 1) + [8], dtype=torch.int64, device=rt.device)
        source = torch.zeros(4096, dtype=torch.uint8, device=rt.device)
        # The kernel stops at once, so the launch's own health check may already report it.
        with pytest.raises(RuntimeError, match="unusable byte count for rank"):
            prepared.gather(source, counts)
            torch.cuda.synchronize()
            rt.check_health()
        assert rt.poisoned and rt.stats()["epoch"] == epoch_before
        assert rt.stats()["error_peer"] == world - 1
    finally:
        rt.close()
        dist.barrier()


def test_gather_fail_stop_when_a_peer_never_sends():
    """Peers of a rank whose proxy stopped time out and poison; that rank fails on its next op."""
    rank = dist.get_rank()
    prepared = _prepare(spin_limit=2_000_000)
    rt = prepared.runtime
    try:
        counts = torch.full((dist.get_world_size(),), CHUNK, dtype=torch.int64, device=rt.device)
        source = torch.ones(CHUNK, dtype=torch.uint8, device=rt.device)
        dist.barrier()
        if rank == 1:
            rt._proxy.stop()
        if rank != 1:
            with pytest.raises(RuntimeError, match="poisoned|timed out"):
                prepared.gather(source, counts)
                torch.cuda.synchronize()
                rt.check_health()
        else:
            # Rank 1 received everything; its poisoned peers never send again.
            prepared.gather(source, counts)
            torch.cuda.synchronize()
            rt.check_health()
            with pytest.raises(RuntimeError, match="poisoned|timed out"):
                prepared.gather(source, counts)
                torch.cuda.synchronize()
                rt.check_health()
        assert rt.poisoned
    finally:
        rt.close()
        dist.barrier()


def test_external_region_describes_the_variable_length_area():
    """A caller-supplied region carries the variable-length area, and the description names it."""
    world, rank = dist.get_world_size(), dist.get_rank()
    supplied = []

    def allocate(nbytes):
        region = torch.full((nbytes,), 0xA5, dtype=torch.uint8, pin_memory=True)
        supplied.append(region)
        return region

    prepared = _prepare(region_allocator=allocate)
    rt = prepared.runtime
    try:
        description = rt.native_description()
        assert description["varlen_bytes"] == CAPACITY
        assert description["region"]["nbytes"] == rt._layout.total_bytes
        counts = [16 * (500 + r) for r in range(world)]
        source = _contribution(rank, counts[rank], 300, rt.device)
        out = torch.full((world, CAPACITY), SENTINEL, dtype=torch.uint8, device=rt.device)
        prepared.gather(source, torch.tensor(counts, dtype=torch.int64, device=rt.device), out=out)
        torch.cuda.synchronize()
        _check(out, counts, 300)
    finally:
        rt.close()
        dist.barrier()
