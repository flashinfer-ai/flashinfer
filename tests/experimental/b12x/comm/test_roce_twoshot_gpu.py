"""Correctness tests for RoCEnante's pipelined two-shot all-reduce.

Run with torchrun on two or more nodes that share a RoCE fabric, for example
from every node of a DGX Spark cluster::

    NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 NCCL_IB_GID_INDEX=3 \\
    torchrun --nnodes=4 --nproc-per-node=1 --node-rank=$RANK \\
        --master-addr=$MASTER --master-port=29650 \\
        -m pytest -x tests/experimental/b12x/comm/test_roce_twoshot_gpu.py

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

ONESHOT_MAX = 1 << 20
PIPE_BYTES = 8 << 20
DTYPES = ("bfloat16", "float32", "float16")


class _Prepared:
    """A runtime plus its published plan; ``all_reduce`` routes by the runtime's thresholds."""

    def __init__(self, runtime, plan):
        self.runtime, self.plan = runtime, plan

    def all_reduce(self, inp, **kwargs):
        return self.runtime.all_reduce(inp, plan=self.plan, **kwargs)

    def oneshot(self, inp, **kwargs):
        """Force the one-shot path (``inp`` must fit ``max_size``)."""
        threshold, self.runtime.pipe_threshold = self.runtime.pipe_threshold, 1 << 62
        try:
            return self.all_reduce(inp, **kwargs)
        finally:
            self.runtime.pipe_threshold = threshold

    def twoshot(self, inp, **kwargs):
        """Force the two-shot path."""
        threshold, self.runtime.pipe_threshold = self.runtime.pipe_threshold, 1
        try:
            return self.all_reduce(inp, **kwargs)
        finally:
            self.runtime.pipe_threshold = threshold


def _prepare(spin_limit: int | None = None, **kwargs) -> _Prepared:
    """A prepared two-shot-capable runtime over the torchrun world."""
    from b12x.comm import roce
    from b12x.comm.roce import _preparation

    if not roce.is_supported():
        pytest.skip(
            "RoCE all-reduce needs an integrated GPU with an active RDMA device"
        )
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
            max_size=ONESHOT_MAX,
            max_gather_bytes=1 << 20,
            pipe_bytes=PIPE_BYTES,
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
        surface="AllReduce.all_reduce",
        call={"dtypes": DTYPES},
        topology="roce_rdma",
        peer_hosts=tuple(f"rank-{r}" for r in range(rt.world_size)),
    )
    plan = roce.plan(query, runtime=rt)
    seeds = [
        torch.zeros(
            16 // getattr(torch, d).itemsize, dtype=getattr(torch, d), device=device
        )
        for d in DTYPES
    ]

    def prepare(state):
        calls = [_preparation.prepared_call(state, inp=seed) for seed in seeds]
        return PreparedCall(run=lambda: [call.run() for call in calls])

    with PreparationSession(
        device=device, autotune=False, compile_workers=2
    ) as session:
        session.prepare((plan.request(name="roce", prepare_call=prepare),))
    return _Prepared(rt, plan)


@pytest.fixture(scope="module")
def runtime():
    prepared = _prepare()
    yield prepared
    prepared.runtime.close()
    dist.barrier()


def _tolerance(dtype: torch.dtype, world: int) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5 * world
    if dtype == torch.bfloat16:
        return 1e-2, 2e-2 * world
    return 2e-3, 4e-3 * world


def _rank_identical(out: torch.Tensor) -> None:
    gathered = [torch.empty_like(out) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, out)
    for peer_out in gathered:
        assert torch.equal(peer_out, out), "ranks must produce bit-identical output"


@pytest.mark.parametrize("dtype", [getattr(torch, d) for d in DTYPES])
# 16 bytes leaves most ranks an empty shard; 16000 and 1 MiB + 48 split unevenly
# across three ranks; 5 MiB spans every chunk flag of a 128 KiB-chunk shard on two.
@pytest.mark.parametrize(
    "nbytes", [16, 16000, 256 * 1024, (1 << 20) + 48, 5 << 20, PIPE_BYTES]
)
def test_twoshot_matches_nccl_and_oneshot(runtime, dtype, nbytes):
    """Two-shot matches NCCL, is rank-identical, and is bitwise one-shot's result."""
    world, rank = dist.get_world_size(), dist.get_rank()
    numel = nbytes // dtype.itemsize
    for trial in range(3):
        torch.manual_seed(1000 * trial + rank)
        inp = torch.randn(numel, dtype=dtype, device=runtime.runtime.device)
        expected = inp.clone()
        dist.all_reduce(expected)
        out = runtime.twoshot(inp)
        torch.cuda.synchronize()
        rtol, atol = _tolerance(dtype, world)
        torch.testing.assert_close(out, expected, rtol=rtol, atol=atol)
        _rank_identical(out)
        if nbytes <= ONESHOT_MAX:
            # Both reduce in FP32 in rank order and round once.
            assert torch.equal(runtime.oneshot(inp), out)
    runtime.runtime.check_health()


def test_size_routing(runtime):
    """Messages route by size alone: one-shot below the threshold, two-shot from it and above max_size."""
    rt = runtime.runtime
    below = min(rt.pipe_threshold, rt.max_size + 16) - 16
    assert not rt._uses_pipe(below)
    assert rt._uses_pipe(min(rt.pipe_threshold, rt.max_size + 16))
    assert rt._uses_pipe(rt.max_size + 16)
    assert not rt._uses_pipe(PIPE_BYTES + 16)
    assert rt.should_allreduce(
        torch.empty(PIPE_BYTES // 2, dtype=torch.bfloat16, device=rt.device)
    )
    assert not rt.should_allreduce(
        torch.empty(PIPE_BYTES // 2 + 8, dtype=torch.bfloat16, device=rt.device)
    )
    for nbytes in (16, 16000, 1 << 20, PIPE_BYTES):
        chunk_packs, chunks = rt._pipe_geometry(nbytes)
        shard = -(-(nbytes // 16) // rt.world_size)
        assert 1 <= chunks <= rt._layout.max_chunks and chunk_packs * chunks >= shard


def test_graph_interleaves_oneshot_and_twoshot(runtime):
    """One graph alternating both paths replays correctly; the slot records switch kinds every op."""
    world, rank = dist.get_world_size(), dist.get_rank()
    rt = runtime.runtime
    torch.manual_seed(77 + rank)
    small = torch.randn(32 * 1024, dtype=torch.bfloat16, device=rt.device)
    large = torch.randn(3 << 19, dtype=torch.bfloat16, device=rt.device)
    small_out, large_out = torch.empty_like(small), torch.empty_like(large)
    expected_small, expected_large = small.clone(), large.clone()
    dist.all_reduce(expected_small)
    dist.all_reduce(expected_large)
    stream = torch.cuda.Stream(device=rt.device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        runtime.oneshot(small, out=small_out)
        runtime.twoshot(large, out=large_out)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream), rt.capture(stream=stream):
        for _ in range(3):
            runtime.oneshot(small, out=small_out)
            runtime.twoshot(large, out=large_out)
            runtime.twoshot(large, out=large_out)
            runtime.oneshot(small, out=small_out)
    for _ in range(5):
        small_out.zero_()
        large_out.zero_()
        dist.barrier()
        graph.replay()
        torch.cuda.synchronize()
        rt.check_health()
        rtol, atol = _tolerance(torch.bfloat16, world)
        torch.testing.assert_close(small_out, expected_small, rtol=rtol, atol=atol)
        torch.testing.assert_close(large_out, expected_large, rtol=rtol, atol=atol)
        _rank_identical(large_out)
    dist.barrier()


def test_proxy_catches_up_on_mixed_doorbells(runtime):
    """A proxy that misses a one-shot doorbell and the two-shot doorbell after it posts both.

    Rank 1's one-shot kernel completes on its peers' payloads, so its two-shot
    kernel rings before the stopped proxy saw the first op; the restarted proxy
    must read each slot's own kind.
    """
    world, rank = dist.get_world_size(), dist.get_rank()
    rt = runtime.runtime
    torch.manual_seed(900 + rank)
    first = torch.randn(4096, dtype=torch.bfloat16, device=rt.device)
    second = torch.randn(1 << 20, dtype=torch.bfloat16, device=rt.device)
    expected = [first.clone(), second.clone()]
    for e in expected:
        dist.all_reduce(e)
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 1:
        rt._proxy.stop()
    outs = [runtime.oneshot(first), runtime.twoshot(second)]
    if rank == 1:
        posted = rt._proxy.stats()["last_seq"]
        deadline = time.monotonic() + 20
        while int(rt._ctrl_words[0].item()) != posted + 2:
            assert time.monotonic() < deadline, "two-shot doorbell never rang"
            time.sleep(0.001)
        rt._proxy.start()
    torch.cuda.synchronize()
    rt.check_health()
    rtol, atol = _tolerance(torch.bfloat16, world)
    for out, exp in zip(outs, expected, strict=True):
        torch.testing.assert_close(out, exp, rtol=rtol, atol=atol)
    dist.barrier()


def test_twoshot_fail_stop_on_timeout(runtime):
    """A two-shot wait that times out poisons every rank and nobody stores a stale reduction.

    Rank 1 stops its proxy for good: the peers time out on its reduce-scatter
    chunks and, being poisoned, publish no all-gather chunk, so rank 1 times
    out on the all-gather instead of finishing with unreduced data.
    """
    rank = dist.get_rank()
    prepared = _prepare(spin_limit=2_000_000)
    rt = prepared.runtime
    epoch_before = rt.stats()["epoch"]
    x = torch.ones(1 << 20, dtype=torch.bfloat16, device=rt.device)
    dist.barrier()
    if rank == 1:
        rt._proxy.stop()
    out = prepared.twoshot(x)
    torch.cuda.synchronize()
    with pytest.raises(RuntimeError, match="poisoned|proxy failed"):
        rt.check_health()
    assert rt.poisoned
    assert rt.stats()["epoch"] == epoch_before
    assert not torch.equal(out, x * dist.get_world_size())
    rt.close()  # teardown during failure must not hang
    dist.barrier()
