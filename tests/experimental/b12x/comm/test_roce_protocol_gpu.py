"""Exercise shared-buffer lifetimes under changing RoCE collective geometry.

Run under torchrun with one GPU process on each of two or four hosts. Both
neighbor-ring and direct-mesh transports use real CuTe kernels and native
RDMA proxies. Gloo carries setup barriers; reduction references are exact
integer sums, so no NCCL communicator is needed.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from tests.experimental.b12x.comm.test_roce_varlen_gpu import (
    CAPACITY,
    SENTINEL,
    _check,
    _contribution,
    _counts_cases,
    _prepare,
)

pytestmark = pytest.mark.skipif(
    not 2 <= int(os.environ.get("WORLD_SIZE", "1")) <= 16,
    reason="requires torchrun with 2..16 ranks",
)


@pytest.fixture(scope="module", params=[False, True], ids=["mesh", "ring"])
def prepared(request):
    if not dist.is_initialized():
        dist.init_process_group("gloo", timeout=timedelta(seconds=120))
    result = _prepare(ring=request.param, pipe_bytes=16 << 20)
    yield result
    dist.barrier()
    result.runtime.close()


def _check_reductions(outputs, iteration):
    """Report corrupt payloads on every rank before a teardown barrier."""
    world = dist.get_world_size()
    errors = []
    for operation, output in enumerate(outputs):
        value = world * (world + 1) // 2 + world * (iteration + operation)
        mismatches = int(torch.count_nonzero(output != value).item())
        if mismatches:
            errors.append(
                {
                    "rank": dist.get_rank(),
                    "replay": iteration,
                    "operation": operation,
                    "nbytes": output.numel() * output.element_size(),
                    "expected": value,
                    "mismatched_elements": mismatches,
                }
            )
    peer_errors = [None] * world
    dist.all_gather_object(peer_errors, errors)
    assert not any(peer_errors), f"collective payload corruption: {peer_errors}"


def _replay(prepared, graph, iteration, *, delay_proxy):
    """Delay one proxy's service independently of GPU and peer progress."""
    runtime = prepared.runtime
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.synchronize()
    dist.barrier()
    resume = None
    if delay_proxy and rank == iteration % world:
        runtime._proxy.stop()

        def restart():
            time.sleep(0.025)
            runtime._proxy.start()

        resume = threading.Thread(target=restart, daemon=True)
        resume.start()
    try:
        # Submission skew changes which GPU reaches each doorbell first.
        time.sleep(0.001 * ((rank + iteration) % world))
        graph.replay()
        torch.cuda.synchronize()
    finally:
        if resume is not None:
            resume.join(timeout=5)
            assert not resume.is_alive(), "delayed proxy did not restart"
    runtime.check_health()


def test_graph_changes_two_shot_shards_before_send_buffer_reuse(prepared):
    """A shared pipe survives 2, 8, 4, and 16 MiB sends within one graph.

    Every operation has a distinct payload. Changing shard boundaries makes
    the next operation overwrite bytes owned by a different rank in the
    preceding all-gather; pipe completion must precede that overwrite.
    """
    runtime = prepared.runtime
    rank = dist.get_rank()
    sizes = (2 << 20, 8 << 20, 4 << 20, 16 << 20, 8 << 20, 2 << 20)
    sources = [
        torch.full(
            (size // 2,), rank + 1 + op, dtype=torch.bfloat16, device=runtime.device
        )
        for op, size in enumerate(sizes)
    ]
    outputs = [torch.empty_like(source) for source in sources]
    stream = torch.cuda.Stream(device=runtime.device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for source, output in zip(sources, outputs, strict=True):
            prepared.all_reduce(source, out=output)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream), runtime.capture(stream=stream):
        for source, output in zip(sources, outputs, strict=True):
            prepared.all_reduce(source, out=output)
    for iteration in range(20):
        for operation, (source, output) in enumerate(
            zip(sources, outputs, strict=True)
        ):
            source.fill_(rank + 1 + iteration + operation)
            output.fill_(float("nan"))
        _replay(prepared, graph, iteration, delay_proxy=iteration % 3 == 0)
        _check_reductions(outputs, iteration)
    dist.barrier()


def test_graph_mixes_variable_counts_and_reduction_with_proxy_delays(prepared):
    """Mutable count vectors and distinct reductions retain their own epochs."""
    runtime = prepared.runtime
    world, rank = dist.get_world_size(), dist.get_rank()
    source = torch.zeros(CAPACITY, dtype=torch.uint8, device=runtime.device)
    counts = torch.zeros(world, dtype=torch.int64, device=runtime.device)
    gathered = torch.empty((world, CAPACITY), dtype=torch.uint8, device=runtime.device)
    inputs = [
        torch.full(
            (size // 2,), rank + 1 + op, dtype=torch.bfloat16, device=runtime.device
        )
        for op, size in enumerate((32768, 8 << 20, 2 << 20))
    ]
    outputs = [torch.empty_like(inp) for inp in inputs]
    stream = torch.cuda.Stream(device=runtime.device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        prepared.gather(source, counts, out=gathered)
        for inp, out in zip(inputs, outputs, strict=True):
            prepared.all_reduce(inp, out=out)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream), runtime.capture(stream=stream):
        prepared.all_reduce(inputs[0], out=outputs[0])
        prepared.gather(source, counts, out=gathered)
        prepared.all_reduce(inputs[1], out=outputs[1])
        prepared.gather(source, counts, out=gathered)
        prepared.all_reduce(inputs[2], out=outputs[2])
    for iteration, values in enumerate(_counts_cases(world) * 3):
        counts.copy_(torch.tensor(values, dtype=torch.int64))
        source.copy_(_contribution(rank, values[rank], 700 + iteration, runtime.device))
        gathered.fill_(SENTINEL)
        for operation, (inp, out) in enumerate(zip(inputs, outputs, strict=True)):
            inp.fill_(rank + 1 + iteration + operation)
            out.fill_(float("nan"))
        _replay(prepared, graph, iteration, delay_proxy=iteration % 3 == 0)
        _check(gathered, values, 700 + iteration)
        _check_reductions(outputs, iteration)
    dist.barrier()
