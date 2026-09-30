"""Correctness tests for the RoCE one-shot all-reduce.

Run with torchrun on two or more nodes that share a RoCE fabric, for example
from every node of a DGX Spark cluster::

    NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 NCCL_IB_GID_INDEX=3 \\
    torchrun --nnodes=4 --nproc-per-node=1 --node-rank=$RANK \\
        --master-addr=$MASTER --master-port=29650 \\
        -m pytest -x tests/comm/test_roce_oneshot_gpu.py

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


class _PreparedRuntime:
    def __init__(self, runtime, plan):
        self._runtime = runtime
        self._plan = plan

    def all_reduce(self, inp, **kwargs):
        return self._runtime.all_reduce(inp, plan=self._plan, **kwargs)

    def all_gather(self, inp, **kwargs):
        return self._runtime.all_gather(inp, plan=self._plan, **kwargs)

    def __getattr__(self, name):
        return getattr(self._runtime, name)


@pytest.fixture(scope="module")
def runtime():
    """Module-scoped RoCEnante runtime over the torchrun world; skips without RDMA support."""
    from b12x.comm import roce
    from b12x.comm.roce import _preparation

    if not roce.is_supported():
        pytest.skip(
            "RoCE all-reduce needs an integrated GPU with an active RDMA device"
        )
    if not dist.is_initialized():
        # A short timeout turns a rank that failed early into an error on every
        # rank instead of a silent hang in the next collective.
        dist.init_process_group("nccl", timeout=timedelta(seconds=60))
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    rt = roce.AllReduce.from_exchange_group(
        exchange_group=dist.group.WORLD,
        device=device,
        max_size=1 << 20,
        max_gather_bytes=4 << 20,
    )
    query = roce.query_from_runtime(
        rt,
        surface="AllReduce.all_reduce",
        call={"dtypes": ("bfloat16", "float32", "float16")},
        topology="roce_rdma",
        peer_hosts=tuple(f"rank-{rank}" for rank in range(rt.world_size)),
    )
    declaration = roce.plan(query, runtime=rt)
    seeds = [torch.zeros(16 // dtype.itemsize, dtype=dtype, device=device)
             for dtype in (torch.float16, torch.bfloat16, torch.float32)]

    def prepare(state):
        calls = [_preparation.prepared_call(state, inp=seed) for seed in seeds]
        calls.append(_preparation.prepared_gather_call(state, inp=seeds[1]))
        return PreparedCall(run=lambda: [call.run() for call in calls])

    request = declaration.request(
        name="roce",
        prepare_call=prepare,
    )
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare((request,))
        yield _PreparedRuntime(rt, declaration)
    rt.close()
    dist.barrier()


def _tolerance(dtype: torch.dtype, world: int) -> tuple[float, float]:
    """(rtol, atol) for a ``world``-way sum in ``dtype`` with float32 accumulation."""
    if dtype == torch.float32:
        return 1e-5, 1e-5 * world
    if dtype == torch.bfloat16:
        return 1e-2, 2e-2 * world
    return 2e-3, 4e-3 * world


def test_payload_is_striped_across_every_hca(runtime):
    """One peer payload is split evenly across both QSFP PCIe functions."""
    if len(runtime.hca_names) < 2:
        pytest.skip("requires two RoCE interfaces for one QSFP port")
    world = dist.get_world_size()
    nbytes = 256 * 1024
    before = runtime.stats()
    before_bytes = before["bytes_posted_per_hca"]
    before_writes = before["writes_completed_per_hca"]
    inp = torch.full(
        (nbytes // 2,), dist.get_rank() + 1, dtype=torch.bfloat16, device=runtime.device
    )
    out = runtime.all_reduce(inp)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, torch.full_like(out, world * (world + 1) // 2))

    expected_bytes = (world - 1) * nbytes // len(runtime.hca_names)
    expected_writes = world - 1
    deadline = time.monotonic() + 5
    while True:
        after = runtime.stats()
        byte_deltas = [
            current - previous
            for current, previous in zip(
                after["bytes_posted_per_hca"], before_bytes, strict=True
            )
        ]
        write_deltas = [
            current - previous
            for current, previous in zip(
                after["writes_completed_per_hca"], before_writes, strict=True
            )
        ]
        if all(delta >= expected_writes for delta in write_deltas):
            break
        assert time.monotonic() < deadline, after
        time.sleep(0.001)
    assert byte_deltas == [expected_bytes] * len(runtime.hca_names)
    assert all(delta >= expected_writes for delta in write_deltas)
    assert after["stripe_hcas"] == list(range(len(runtime.hca_names)))
    dist.barrier()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float16])
@pytest.mark.parametrize("numel_bytes", [16, 4096, 48 * 1024, 256 * 1024, 1 << 20])
def test_matches_nccl_and_is_rank_identical(runtime, dtype, numel_bytes):
    """All-reduce matches NCCL within tolerance and is bit-identical across ranks."""
    world = dist.get_world_size()
    rank = dist.get_rank()
    numel = numel_bytes // torch.tensor([], dtype=dtype).element_size()
    for trial in range(4):
        torch.manual_seed(1000 * trial + rank)
        inp = torch.randn(numel, dtype=dtype, device=runtime.device)
        expected = inp.clone()
        dist.all_reduce(expected)
        out = runtime.all_reduce(inp)
        torch.cuda.synchronize()
        rtol, atol = _tolerance(dtype, world)
        if not torch.allclose(out, expected, rtol=rtol, atol=atol):
            print(
                f"[rank {rank} trial {trial} {dtype} {numel_bytes}B] MISMATCH stats={runtime.stats()}\n"
                f"  inp={inp[:8].tolist()}\n  out={out[:8].tolist()}\n  exp={expected[:8].tolist()}",
                flush=True,
            )
        torch.testing.assert_close(out, expected, rtol=rtol, atol=atol)
        gathered = [torch.empty_like(out) for _ in range(world)]
        dist.all_gather(gathered, out)
        for peer_out in gathered:
            assert torch.equal(peer_out, out), "ranks must produce bit-identical output"


def test_rejects_ineligible_inputs(runtime):
    """Size, shape, stride, and pointer-alignment boundaries of all-reduce eligibility."""
    huge = torch.zeros(
        (runtime.max_size // 2) + 8, dtype=torch.bfloat16, device=runtime.device
    )
    assert not runtime.should_allreduce(huge)
    odd = torch.zeros(3, dtype=torch.bfloat16, device=runtime.device)
    assert not runtime.should_allreduce(odd)
    strided = torch.zeros(64, 8, dtype=torch.bfloat16, device=runtime.device)[:, ::2]
    assert not runtime.should_allreduce(strided)
    ok = torch.zeros(4096, dtype=torch.bfloat16, device=runtime.device)
    assert runtime.should_allreduce(ok)
    # Eligibility is rank-invariant: a pointer that is not 16-byte aligned is
    # still eligible and is staged through scratch, for input and for output.
    rank = dist.get_rank()
    torch.manual_seed(4242 + rank)
    backing = torch.randn(4097, dtype=torch.float16, device=runtime.device)
    unaligned = backing[1:]
    assert unaligned.is_contiguous() and unaligned.data_ptr() % 16 != 0
    assert runtime.should_allreduce(unaligned)
    expected = unaligned.clone()
    dist.all_reduce(expected)
    rtol, atol = _tolerance(torch.float16, dist.get_world_size())
    torch.testing.assert_close(
        runtime.all_reduce(unaligned), expected, rtol=rtol, atol=atol
    )
    out_backing = torch.empty(4097, dtype=torch.float16, device=runtime.device)
    aligned_in = unaligned.clone()
    torch.testing.assert_close(
        runtime.all_reduce(aligned_in, out=out_backing[1:]),
        expected,
        rtol=rtol,
        atol=atol,
    )
    # a foreign-device output never reaches the kernel
    with pytest.raises(ValueError):
        runtime.all_reduce(aligned_in, out=torch.empty(4096, dtype=torch.float16))


def test_cuda_graph_replay(runtime):
    """Three captured all-reduces replay correctly with the epoch advancing in-graph."""
    world = dist.get_world_size()
    rank = dist.get_rank()
    static_in = torch.zeros(24 * 1024, dtype=torch.bfloat16, device=runtime.device)
    static_out = torch.empty_like(static_in)
    stream = torch.cuda.Stream(device=runtime.device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            runtime.all_reduce(static_in, out=static_out)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream), runtime.capture(stream=stream):
        for _ in range(3):
            runtime.all_reduce(static_in, out=static_out)
            static_in.copy_(static_out)
    torch.cuda.synchronize()
    dist.barrier()
    for replay in range(5):
        torch.manual_seed(7 * replay + rank)
        seed = torch.randn_like(static_in) * 0.01
        static_in.copy_(seed)
        expected = seed.clone()
        for _ in range(3):
            dist.all_reduce(expected)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            static_out, expected, rtol=2e-2, atol=2e-2 * world**3
        )


def test_cuda_graph_replay_with_alternating_grid_sizes(runtime):
    """Small and MTP-sized reductions may alternate without sharing arrivals."""
    world = dist.get_world_size()
    rank = dist.get_rank()
    small = torch.zeros(4096, dtype=torch.bfloat16, device=runtime.device)
    mtp = torch.zeros(32768, dtype=torch.bfloat16, device=runtime.device)
    small_out = torch.empty_like(small)
    mtp_out = torch.empty_like(mtp)
    stream = torch.cuda.Stream(device=runtime.device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        runtime.all_reduce(small, out=small_out)
        runtime.all_reduce(mtp, out=mtp_out)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream), runtime.capture(stream=stream):
        runtime.all_reduce(small, out=small_out)
        runtime.all_reduce(mtp, out=mtp_out)
        runtime.all_reduce(small, out=small_out)
    torch.cuda.synchronize()
    dist.barrier()
    rtol, atol = _tolerance(torch.bfloat16, world)
    for replay in range(24):
        torch.manual_seed(101 * replay + rank)
        small.copy_(torch.randn_like(small))
        mtp.copy_(torch.randn_like(mtp))
        expected_small = small.clone()
        expected_mtp = mtp.clone()
        dist.all_reduce(expected_small)
        dist.all_reduce(expected_mtp)
        graph.replay()
        torch.cuda.synchronize()
        runtime.check_health()
        torch.testing.assert_close(small_out, expected_small, rtol=rtol, atol=atol)
        torch.testing.assert_close(mtp_out, expected_mtp, rtol=rtol, atol=atol)


def test_proxy_catches_up_after_missed_doorbell(runtime):
    """A proxy that was descheduled across two doorbells must post both ops.

    A rank's kernel for op N completes on the peers' payloads alone, so op N+1
    can ring the doorbell before the proxy has seen op N.  Rank 1 stops its
    proxy thread, launches two all-reduces, and restarts the thread only once
    the second doorbell has overwritten the first; the peers wait on rank 1's
    first payload in the meantime.
    """
    world = dist.get_world_size()
    rank = dist.get_rank()
    inputs, expected = [], []
    for i in range(2):
        torch.manual_seed(900 + 10 * i + rank)
        x = torch.randn(8192, dtype=torch.bfloat16, device=runtime.device)
        e = x.clone()
        dist.all_reduce(e)
        inputs.append(x)
        expected.append(e)
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 1:
        runtime._proxy.stop()
    outs = [runtime.all_reduce(x) for x in inputs]
    if rank == 1:
        posted = runtime._proxy.stats()["last_seq"]
        deadline = time.monotonic() + 20
        # Poll the host-side control word; ``runtime.stats()`` reads the device
        # epoch and would synchronize the stream, i.e. wait for the stuck op.
        while int(runtime._ctrl_words[0].item()) != posted + 2:
            assert time.monotonic() < deadline, "second doorbell never rang"
            time.sleep(0.001)
        runtime._proxy.start()
    torch.cuda.synchronize()
    runtime.check_health()
    rtol, atol = _tolerance(torch.bfloat16, world)
    for out, exp in zip(outs, expected, strict=True):
        torch.testing.assert_close(out, exp, rtol=rtol, atol=atol)
    dist.barrier()


def test_alternating_eager_streams_are_ordered(runtime):
    """Collectives issued on two alternating streams execute in launch order.

    The runtime has one epoch and two transport slots, so it chains a launch
    on a different stream behind the previous one with an event; without that
    the two streams could run two collectives concurrently.
    """
    world = dist.get_world_size()
    rank = dist.get_rank()
    torch.manual_seed(77 + rank)
    x = torch.randn(48 * 1024, dtype=torch.bfloat16, device=runtime.device)
    expected = x.clone()
    dist.all_reduce(expected)
    streams = (
        torch.cuda.Stream(device=runtime.device),
        torch.cuda.Stream(device=runtime.device),
    )
    for s in streams:
        s.wait_stream(torch.cuda.current_stream())
    outs = []
    for i in range(8):
        with torch.cuda.stream(streams[i % 2]):
            outs.append(runtime.all_reduce(x * (i + 1)))
    torch.cuda.synchronize()
    runtime.check_health()
    rtol, atol = _tolerance(torch.bfloat16, world)
    for i, out in enumerate(outs):
        torch.testing.assert_close(
            out, expected * (i + 1), rtol=rtol, atol=atol * (i + 1)
        )
    dist.barrier()


def test_adapter_path_graph_replay(runtime):
    """The exact vLLM adapter calls, captured once and replayed many times.

    ``all_reduce(inp)`` and ``all_gather(inp, dim=-1)`` without ``out``, a
    direct-layout gather, a padded gather, and mixed ordering.  Outputs are
    allocated inside the capture; every replay must land in the same tensors
    (stable addresses) with the right values, and the runtime must stay healthy.
    """
    world = dist.get_world_size()
    rank = dist.get_rank()
    h = torch.zeros(6 * 4096, dtype=torch.bfloat16, device=runtime.device)
    logits = torch.zeros(6, 38720, dtype=torch.bfloat16, device=runtime.device)
    odd = torch.zeros(
        5, 3, dtype=torch.bfloat16, device=runtime.device
    )  # 30-byte rows: padded path
    stream = torch.cuda.Stream(device=runtime.device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        runtime.all_reduce(h)
        runtime.all_gather(logits, dim=-1)
        runtime.all_gather(odd, dim=-1)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream), runtime.capture(stream=stream):
        r1 = runtime.all_reduce(h)
        g1 = runtime.all_gather(logits, dim=-1)
        r2 = runtime.all_reduce(r1)
        g2 = runtime.all_gather(odd, dim=-1)
        r3 = runtime.all_reduce(r2)
    torch.cuda.synchronize()
    dist.barrier()
    addresses = (
        r1.data_ptr(),
        g1.data_ptr(),
        r2.data_ptr(),
        g2.data_ptr(),
        r3.data_ptr(),
    )
    rtol, atol = _tolerance(torch.bfloat16, world)
    for replay in range(24):
        torch.manual_seed(31 * replay + rank)
        h.copy_(torch.randn_like(h) * 0.01)
        logits.copy_(torch.randn_like(logits))
        odd.copy_(torch.randn_like(odd))
        e1 = h.clone()
        dist.all_reduce(e1)
        e2 = e1.clone()
        dist.all_reduce(e2)
        e3 = e2.clone()
        dist.all_reduce(e3)
        parts = [torch.empty_like(logits) for _ in range(world)]
        dist.all_gather(parts, logits)
        eg1 = torch.cat(parts, dim=-1)
        parts = [torch.empty_like(odd) for _ in range(world)]
        dist.all_gather(parts, odd)
        eg2 = torch.cat(parts, dim=-1)
        graph.replay()
        torch.cuda.synchronize()
        runtime.check_health()
        assert (
            r1.data_ptr(),
            g1.data_ptr(),
            r2.data_ptr(),
            g2.data_ptr(),
            r3.data_ptr(),
        ) == addresses
        torch.testing.assert_close(r1, e1, rtol=rtol, atol=atol)
        torch.testing.assert_close(r2, e2, rtol=rtol, atol=atol * world)
        torch.testing.assert_close(r3, e3, rtol=rtol, atol=atol * world**2)
        assert torch.equal(g1, eg1)
        assert torch.equal(g2, eg2)
    dist.barrier()


def _fresh_runtime(spin_limit: int):
    """A runtime of its own with a short spin limit, for fault injection."""
    from b12x.comm import roce

    previous = os.environ.get("B12X_ROCE_SPIN_LIMIT")
    os.environ["B12X_ROCE_SPIN_LIMIT"] = str(spin_limit)
    try:
        rt = roce.AllReduce.from_exchange_group(
            exchange_group=dist.group.WORLD,
            device=torch.device("cuda", 0),
            max_size=1 << 20,
        )
    finally:
        if previous is None:
            del os.environ["B12X_ROCE_SPIN_LIMIT"]
        else:
            os.environ["B12X_ROCE_SPIN_LIMIT"] = previous
    rt.prepare((torch.bfloat16,))
    return rt


def test_fail_stop_on_timeout_eager(runtime):
    """A wait that times out poisons the runtime: the epoch stops, the failure
    is raised on the next check, another launch raises before enqueue, and the
    peers converge to the same state on their own.

    Rank 1 stops its proxy for good.  Every other rank's first collective times
    out (asymmetric: rank 1's own kernel completes on the peers' payloads).
    Rank 1 then times out on its second collective because the poisoned peers
    launch no-ops and never post again, so no rank can proceed alone.
    """
    world = dist.get_world_size()
    rank = dist.get_rank()
    rt = _fresh_runtime(2_000_000)
    epoch_before = rt.stats()["epoch"]
    dist.barrier()
    if rank == 1:
        rt._proxy.stop()
    x = torch.ones(4096, dtype=torch.bfloat16, device=rt.device)
    out = rt.all_reduce(
        x
    )  # enqueue succeeds: the fault is only visible once the kernel waited
    torch.cuda.synchronize()
    if rank != 1:
        with pytest.raises(RuntimeError, match="poisoned"):
            rt.check_health()
        assert rt.poisoned
        assert rt.stats()["epoch"] == epoch_before
        with pytest.raises(RuntimeError, match="poisoned"):
            rt.all_reduce(x)  # fatal, never a fallback
    else:
        rt.check_health()
        assert rt.stats()["epoch"] == epoch_before + 1
        torch.testing.assert_close(out, x * world)
        rt.all_reduce(x)
        torch.cuda.synchronize()
        with pytest.raises(RuntimeError, match="poisoned"):
            rt.check_health()
    assert rt.poisoned
    rt.close()  # teardown during failure must not hang
    dist.barrier()


def test_fail_stop_on_timeout_graph_replay(runtime):
    """A faulted replay leaves the epoch where it was and raises on the post-step check."""
    world = dist.get_world_size()
    rank = dist.get_rank()
    rt = _fresh_runtime(2_000_000)
    static = torch.ones(4096, dtype=torch.bfloat16, device=rt.device)
    stream = torch.cuda.Stream(device=rt.device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        rt.all_reduce(static)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream), rt.capture(stream=stream):
        o1 = rt.all_reduce(static)
        o2 = rt.all_reduce(o1)
        o3 = rt.all_reduce(o2)
    torch.cuda.synchronize()
    dist.barrier()
    graph.replay()
    torch.cuda.synchronize()
    rt.check_health()
    torch.testing.assert_close(o3, static * world**3)
    epoch_ok = rt.stats()["epoch"]
    dist.barrier()
    if rank == 1:
        rt._proxy.stop()
    o3.zero_()
    graph.replay()
    torch.cuda.synchronize()
    with pytest.raises(RuntimeError, match="poisoned"):
        rt.check_health()
    # no rank advanced past the failed sequence by more than the ops that
    # completed on the peers' payloads, and none produced the step's result
    assert rt.stats()["epoch"] < epoch_ok + 3
    assert not torch.equal(o3, static * world**3)
    rt.close()
    dist.barrier()


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float32, torch.int64, torch.int32]
)
@pytest.mark.parametrize(
    "shape,dim",
    [
        ((6, 38720), -1),
        ((6, 38720), 1),
        ((16, 4096), 0),
        ((4096,), 0),
        ((2, 3, 1024), -1),
        ((96, 8192), 1),
        ((6, 8), -1),
        # unaligned rows / sizes: padded contiguous path + torch reshape
        ((6, 2), -1),
        ((5, 2), -1),
        ((7, 3), 0),
        ((6, 1), -1),
        ((3,), 0),
    ],
)
def test_all_gather_matches_torch(runtime, dtype, shape, dim):
    """All-gather equals ``torch.cat`` of NCCL shards on both gather paths."""
    world = dist.get_world_size()
    rank = dist.get_rank()
    for trial in range(3):
        torch.manual_seed(500 * trial + rank)
        if dtype.is_floating_point:
            inp = torch.randn(*shape, dtype=dtype, device=runtime.device)
        else:
            inp = torch.randint(-1000, 1000, shape, dtype=dtype, device=runtime.device)
        if inp.numel() * inp.element_size() > runtime.max_gather_bytes:
            assert not runtime.should_all_gather(inp, dim)
            pytest.skip("shard exceeds the runtime's all-gather capacity")
        assert runtime.should_all_gather(inp, dim)
        gathered = [torch.empty_like(inp) for _ in range(world)]
        dist.all_gather(gathered, inp)
        expected = torch.cat(gathered, dim=dim)
        out = runtime.all_gather(inp, dim=dim)
        torch.cuda.synchronize()
        assert out.shape == expected.shape
        assert torch.equal(out, expected)


def test_all_gather_rejects_ineligible(runtime):
    """Dim, capacity, and ``out`` validation of the all-gather, before any launch."""
    x = torch.zeros(4, 8, 8, dtype=torch.bfloat16, device=runtime.device)
    assert not runtime.should_all_gather(x, 1)  # middle dim
    odd = torch.zeros(6, 5, dtype=torch.bfloat16, device=runtime.device)
    assert runtime.should_all_gather(odd, -1)  # unaligned rows take the padded path
    assert not runtime._direct_gather_layout(odd, 1)
    huge = torch.zeros(
        (runtime.max_gather_bytes // 2) + 8, dtype=torch.bfloat16, device=runtime.device
    )
    assert not runtime.should_all_gather(huge, 0)
    # ``out`` is validated before anything is launched, on both gather paths
    world = dist.get_world_size()
    aligned = torch.zeros(6, 8, dtype=torch.bfloat16, device=runtime.device)
    for shard in (odd, aligned):
        wrong_dtype = torch.empty(
            shard.shape[0],
            shard.shape[1] * world,
            dtype=torch.float32,
            device=runtime.device,
        )
        with pytest.raises(ValueError):
            runtime.all_gather(shard, dim=-1, out=wrong_dtype)
        wrong_shape = torch.empty(
            shard.shape[0], shard.shape[1], dtype=torch.bfloat16, device=runtime.device
        )
        with pytest.raises(ValueError):
            runtime.all_gather(shard, dim=-1, out=wrong_shape)
        with pytest.raises(ValueError):
            runtime.all_gather(
                shard,
                dim=-1,
                out=torch.empty(
                    shard.shape[0], shard.shape[1] * world, dtype=torch.bfloat16
                ),
            )


def test_all_gather_graph_replay_with_alternating_grid_sizes(runtime):
    """Tiny (top-k) and large (logits) gathers may alternate in one graph.

    The gather grid follows the shard size like the all-reduce grid, and each
    power-of-two grid owns its arrival counters, so a one-block gather may sit
    between full-grid gathers and reductions without sharing arrivals.
    """
    world = dist.get_world_size()
    rank = dist.get_rank()
    topk = torch.zeros(8, 64, dtype=torch.float32, device=runtime.device)
    logits = torch.zeros(8, 38720, dtype=torch.bfloat16, device=runtime.device)
    h = torch.zeros(8 * 4096, dtype=torch.bfloat16, device=runtime.device)
    topk_out = torch.empty(8, 64 * world, dtype=torch.float32, device=runtime.device)
    logits_out = torch.empty(
        8, 38720 * world, dtype=torch.bfloat16, device=runtime.device
    )
    reduced = torch.empty_like(h)
    from b12x.comm.roce import roce_oneshot

    small_grid = roce_oneshot._grid_blocks(
        topk.numel() * topk.element_size() // roce_oneshot.PACK_BYTES,
        runtime._threads,
        runtime._blocks,
    )
    large_grid = roce_oneshot._grid_blocks(
        logits.numel() * logits.element_size() // roce_oneshot.PACK_BYTES,
        runtime._threads,
        runtime._blocks,
    )
    assert small_grid == 1
    assert large_grid > small_grid
    stream = torch.cuda.Stream(device=runtime.device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        runtime.all_gather(topk, dim=-1, out=topk_out)
        runtime.all_gather(logits, dim=-1, out=logits_out)
        runtime.all_reduce(h, out=reduced)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream), runtime.capture(stream=stream):
        runtime.all_gather(topk, dim=-1, out=topk_out)
        runtime.all_gather(logits, dim=-1, out=logits_out)
        runtime.all_gather(topk, dim=-1, out=topk_out)
        runtime.all_reduce(h, out=reduced)
        runtime.all_gather(topk, dim=-1, out=topk_out)
    torch.cuda.synchronize()
    dist.barrier()
    for replay in range(24):
        torch.manual_seed(31 * replay + rank)
        topk.copy_(torch.randn_like(topk))
        logits.copy_(torch.randn_like(logits))
        h.copy_(torch.randn_like(h))
        parts = [torch.empty_like(topk) for _ in range(world)]
        dist.all_gather(parts, topk)
        expected_topk = torch.cat(parts, dim=-1)
        parts = [torch.empty_like(logits) for _ in range(world)]
        dist.all_gather(parts, logits)
        expected_logits = torch.cat(parts, dim=-1)
        expected_reduce = h.clone()
        dist.all_reduce(expected_reduce)
        graph.replay()
        torch.cuda.synchronize()
        runtime.check_health()
        assert torch.equal(topk_out, expected_topk)
        assert torch.equal(logits_out, expected_logits)
        torch.testing.assert_close(
            reduced, expected_reduce, rtol=2e-2, atol=2e-2 * world
        )


def test_all_gather_graph_replay_mixed_with_all_reduce(runtime):
    """A captured graph mixing both collectives replays correctly."""
    world = dist.get_world_size()
    rank = dist.get_rank()
    x = torch.zeros(6, 38720, dtype=torch.bfloat16, device=runtime.device)
    h = torch.zeros(6 * 4096, dtype=torch.bfloat16, device=runtime.device)
    gathered = torch.empty(
        6, 38720 * world, dtype=torch.bfloat16, device=runtime.device
    )
    reduced = torch.empty_like(h)
    stream = torch.cuda.Stream(device=runtime.device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        runtime.all_reduce(h, out=reduced)
        runtime.all_gather(x, dim=-1, out=gathered)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream), runtime.capture(stream=stream):
        runtime.all_reduce(h, out=reduced)
        runtime.all_gather(x, dim=-1, out=gathered)
        runtime.all_reduce(h, out=reduced)
    torch.cuda.synchronize()
    dist.barrier()
    for replay in range(4):
        torch.manual_seed(11 * replay + rank)
        x.copy_(torch.randn_like(x))
        h.copy_(torch.randn_like(h))
        parts = [torch.empty_like(x) for _ in range(world)]
        dist.all_gather(parts, x)
        expected_gather = torch.cat(parts, dim=-1)
        expected_reduce = h.clone()
        dist.all_reduce(expected_reduce)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(gathered, expected_gather)
        torch.testing.assert_close(
            reduced, expected_reduce, rtol=2e-2, atol=2e-2 * world
        )
