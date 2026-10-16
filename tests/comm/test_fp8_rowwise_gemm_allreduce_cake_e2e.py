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

Four-GPU numerical test of the Cake fused FP8 row-wise GEMM + all-reduce
backend on an NCCL group of four SM100 devices: one-shot calls with workspace
growth, a caller-provided output, strided-operand rejection, the
capacity-bound prepared launcher, both torch symmetric-memory backends (the
default CUDA backend and NVSHMEM) and CUDA-graph capture with replays
interleaved with eager calls.
"""

import gc
import random
import weakref

import pytest
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.multiprocessing as mp
from flashinfer.comm import (
    fp8_rowwise_gemm_allreduce,
    prepare_fp8_rowwise_gemm_allreduce,
)
from flashinfer.comm.fp8_rowwise_gemm_allreduce import (
    cake_fp8_rowwise_gemm_allreduce as backend,
)
from flashinfer.utils import get_compute_capability

WORLD_SIZE = 4
N = 3072
K_L = 768
FP8 = torch.float8_e4m3fn


def _inputs(m, n, k_l, device, generator):
    """FP8 (e4m3) operands with fp32 row and column scales."""

    a = torch.randn(m, k_l, device=device, generator=generator).to(FP8)
    w = torch.randn(n, k_l, device=device, generator=generator).to(FP8)
    a_scale = 0.5 + torch.rand(m, device=device, generator=generator)
    w_scale = (0.5 + torch.rand(n, device=device, generator=generator)) / k_l
    return a, a_scale, w, w_scale


def _refill(a, a_scale, w, w_scale, generator):
    """Fresh random data in place (static CUDA-graph inputs)."""

    a.copy_(torch.randn(a.shape, device=a.device, generator=generator).to(FP8))
    w.copy_(torch.randn(w.shape, device=w.device, generator=generator).to(FP8))
    a_scale.copy_(0.5 + torch.rand(a_scale.shape, device=a.device, generator=generator))
    w_scale.copy_(
        (0.5 + torch.rand(w_scale.shape, device=a.device, generator=generator))
        / a.shape[1]
    )


def _expected(a, a_scale, w, w_scale, group):
    """fp32 all-reduce of the scaled partial products; returns (bf16 result, sum |P_r|)."""

    partial = (a.float() @ w.float().t()) * a_scale[:, None] * w_scale[None, :]
    total = partial.clone()
    dist.all_reduce(total, op=dist.ReduceOp.SUM, group=group)
    magnitude = partial.abs()
    dist.all_reduce(magnitude, op=dist.ReduceOp.SUM, group=group)
    return total.to(torch.bfloat16), magnitude


def _check(actual, expected):
    """|y - r| <= 1e-2 + 1e-2 * sum_r |P_r| elementwise and relative L2 within 2**-7."""

    reference, magnitude = expected
    assert actual.dtype == torch.bfloat16 and actual.shape == reference.shape
    error = (actual.float() - reference.float()).abs()
    bound = 1e-2 + 1e-2 * magnitude
    assert bool((error <= bound).all()), (
        f"max |error| {error.max().item():.4f} exceeds the bound at "
        f"{int((error > bound).sum())} elements"
    )
    rel_l2 = error.norm() / reference.float().norm().clamp_min(1e-12)
    assert rel_l2.item() <= 2**-7, f"relative L2 {rel_l2.item():.3e}"


def _replayed_launcher(launcher, static, device, group, generator, eager):
    """Capture two launcher calls into one graph, replay with fresh data, interleave eager calls."""

    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(capture_stream):
        warm = launcher(*static)
    torch.cuda.current_stream(device).wait_stream(capture_stream)
    _check(warm, _expected(*static, group))
    del warm
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        captured_first = launcher(*static)
        captured_second = launcher(*static)
    for _replay in range(3):
        _refill(*static, generator)
        graph.replay()
        expected = _expected(*static, group)
        _check(captured_first, expected)
        _check(captured_second, expected)
        _check(launcher(*eager), _expected(*eager, group))
        del expected
    del graph, captured_first, captured_second


def _run_cake_group(rank: int, port: int, symm_backend: str):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    if symm_backend == "NVSHMEM":
        symm_mem.set_backend("NVSHMEM")
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=WORLD_SIZE,
        device_id=device,
    )
    group = dist.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")
    symm_mem.enable_symm_mem_for_group(group.group_name)
    generator = torch.Generator(device=device)
    generator.manual_seed(1188 + rank)

    # One-shot calls: the first sizes the workspace (one collective), 37 rows
    # fit the 128-row capacity, 1024 rows grow it once. Two consecutive calls
    # return fresh outputs and the first is not overwritten.
    for m in (1, 37, 1024):
        a, a_scale, w, w_scale = _inputs(m, N, K_L, device, generator)
        expected = _expected(a, a_scale, w, w_scale, group)
        first = fp8_rowwise_gemm_allreduce(
            a, a_scale, w, w_scale, group, backend="cake"
        )
        assert first.shape == (m, N) and first.dtype == torch.bfloat16
        _check(first, expected)
        first_snapshot = first.clone()
        second = fp8_rowwise_gemm_allreduce(a, a_scale, w, w_scale, group)
        assert first.data_ptr() != second.data_ptr()
        torch.testing.assert_close(first, first_snapshot, atol=0, rtol=0)
        _check(second, expected)
        del first, first_snapshot, second, expected, a, a_scale, w, w_scale
    assert backend.symmetric_memory_backend(device) == symm_backend
    workspace = backend._workspace_of(group, device.index, N)
    assert workspace.rows_capacity == 1024 and workspace.s_bytes == 2 * N * 4 * 1024
    assert workspace.max_clusters > 0 and workspace.captured is False
    if rank == 0:
        print(
            "[cake fp8 gemm+allreduce e2e] "
            f"symm={symm_backend} clusters={workspace.max_clusters} "
            f"flags_in_pad={workspace.flags_in_signal_pad}",
            flush=True,
        )

    # The backend callable writes a caller-provided output in place; strided
    # operands are rejected, not copied.
    a, a_scale, w, w_scale = _inputs(1000, N, K_L, device, generator)
    expected = _expected(a, a_scale, w, w_scale, group)
    out = torch.full((1000, N), float("nan"), dtype=torch.bfloat16, device=device)
    assert (
        backend.fp8_rowwise_gemm_allreduce_cake(
            a, a_scale, w, w_scale, group, backend="cake", out=out
        )
        is out
    )
    _check(out, expected)
    with pytest.raises(ValueError, match="a must be contiguous"):
        fp8_rowwise_gemm_allreduce(
            a.t().contiguous().t(), a_scale, w, w_scale, group, backend="cake"
        )
    with pytest.raises(ValueError, match="w must be contiguous"):
        fp8_rowwise_gemm_allreduce(
            a, a_scale, w.t().contiguous().t(), w_scale, group, backend="cake"
        )
    with pytest.raises(ValueError, match="a_scale must have dtype"):
        fp8_rowwise_gemm_allreduce(
            a, a_scale.to(torch.bfloat16), w, w_scale, group, backend="cake"
        )
    # The backend keeps no reference to the caller's tensors.
    refs = [weakref.ref(t) for t in (a, a_scale, w, w_scale, out)]
    torch.cuda.synchronize(device)
    del a, a_scale, w, w_scale, out, expected
    gc.collect()
    assert all(ref() is None for ref in refs)

    # A call issued on another stream reuses the same workspace and is ordered
    # after the previous call.
    producer_stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(producer_stream):
        a, a_scale, w, w_scale = _inputs(300, N, K_L, device, generator)
        expected = _expected(a, a_scale, w, w_scale, group)
        result = fp8_rowwise_gemm_allreduce(
            a, a_scale, w, w_scale, group, backend="cake"
        )
    torch.cuda.current_stream(device).wait_stream(producer_stream)
    _check(result, expected)
    del a, a_scale, w, w_scale, expected, result

    # Capacity-bound prepared launcher: one collective at preparation, then any
    # row count up to max_rows without another collective; a larger call, a
    # different N or K_l are refused with named errors.
    sample = _inputs(1, N, K_L, device, generator)
    launcher = prepare_fp8_rowwise_gemm_allreduce(*sample, group, max_rows=4133)
    assert launcher.max_rows == 4133 and launcher.n == N and launcher.k_l == K_L
    assert launcher.workspace_bytes == 2 * N * 4 * 4224
    assert launcher.max_active_clusters == workspace.max_clusters
    for served_rows in (1, 129, 4133):
        served = _inputs(served_rows, N, K_L, device, generator)
        served_out = launcher(*served)
        assert served_out.shape == (served_rows, N)
        _check(served_out, _expected(*served, group))
        del served, served_out
    with pytest.raises(ValueError, match=r"\[capacity_exceeded\]"):
        launcher(*_inputs(4134, N, K_L, device, generator))
    with pytest.raises(ValueError, match=r"\[N_mismatch\]"):
        launcher(*_inputs(1, 2 * N, K_L, device, generator))
    with pytest.raises(ValueError, match=r"\[K_l_mismatch\]"):
        launcher(*_inputs(1, N, 2 * K_L, device, generator))
    # The prepared launcher is CUDA-graph capturable: two calls in one graph,
    # three replays with fresh static data, an eager call between replays.
    for static_rows in (37, 1024):
        static = _inputs(static_rows, N, K_L, device, generator)
        _replayed_launcher(launcher, static, device, group, generator, sample)
        del static

    # Growth after capture is refused: a graph captured from a one-shot call
    # bakes the addresses of the current workspace generation on every rank, so
    # a later eager one-shot call needing more rows raises a named error before
    # any collective. The captured graph keeps replaying correctly and the
    # workspace still serves every row count within its capacity.
    graph_static = _inputs(1024, N, K_L, device, generator)
    growth_stream = torch.cuda.Stream(device=device)
    growth_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(growth_stream):
        warm = fp8_rowwise_gemm_allreduce(*graph_static, group, backend="cake")
    torch.cuda.current_stream(device).wait_stream(growth_stream)
    del warm
    growth_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(growth_graph, stream=growth_stream):
        captured_out = fp8_rowwise_gemm_allreduce(*graph_static, group, backend="cake")
    assert backend._workspace_of(group, device.index, N).captured is True
    grown = _inputs(1025, N, K_L, device, generator)
    with pytest.raises(RuntimeError, match="captured into a CUDA graph"):
        fp8_rowwise_gemm_allreduce(*grown, group, backend="cake")
    for _replay in range(2):
        _refill(*graph_static, generator)
        growth_graph.replay()
        _check(captured_out, _expected(*graph_static, group))
    for served_rows in (1, 1000):
        served = _inputs(served_rows, N, K_L, device, generator)
        _check(
            fp8_rowwise_gemm_allreduce(*served, group, backend="cake"),
            _expected(*served, group),
        )
        del served
    del growth_graph, captured_out, graph_static, grown
    # Capture never allocates: a one-shot call for another N inside capture
    # would need a new workspace and raises before any collective.
    other = _inputs(1, 2 * N, K_L, device, generator)
    refusal_stream = torch.cuda.Stream(device=device)
    refusal_graph = torch.cuda.CUDAGraph()
    with (
        pytest.raises(RuntimeError, match="inside CUDA-graph capture"),
        torch.cuda.graph(refusal_graph, stream=refusal_stream),
    ):
        fp8_rowwise_gemm_allreduce(*other, group, backend="cake")
    del refusal_graph, other, launcher, sample

    torch.cuda.synchronize(device)
    dist.destroy_process_group(group)
    dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() != WORLD_SIZE,
    reason="Cake fused FP8 GEMM + all-reduce e2e requires exactly four visible GPUs",
)
@pytest.mark.skipif(
    torch.cuda.device_count() == 0
    or get_compute_capability(torch.device("cuda:0")) != (10, 0),
    reason="Cake fused FP8 GEMM + all-reduce e2e requires SM100",
)
@pytest.mark.parametrize(
    "symm_backend",
    [
        "CUDA",
        pytest.param(
            "NVSHMEM",
            marks=pytest.mark.skipif(
                not symm_mem.is_nvshmem_available(),
                reason="torch symmetric memory has no NVSHMEM backend here",
            ),
        ),
    ],
)
def test_fp8_rowwise_gemm_allreduce_cake_four_ranks(symm_backend):
    port = random.randint(30000, 60000)
    mp.spawn(
        _run_cake_group,
        args=(port, symm_backend),
        nprocs=WORLD_SIZE,
        join=True,
    )
