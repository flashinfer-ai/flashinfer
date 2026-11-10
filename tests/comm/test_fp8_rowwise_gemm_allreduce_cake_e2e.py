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

Multi-GPU numerical test of the Cake fused FP8 row-wise GEMM + all-reduce
backend on an NCCL group of every visible GPU (two, four or eight devices of
one delivered architecture): one-shot calls with workspace growth, a
caller-provided output, strided-operand rejection, the capacity-bound prepared
launcher, both torch symmetric-memory backends (the default CUDA backend and
NVSHMEM) and CUDA-graph capture with replays interleaved with eager calls. The
world size is the visible device count; the carrier (unicast or NVLink
multicast) follows the loader's table for that world size.
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
from flashinfer.jit import cake_fp8_rowwise_gemm_allreduce as loader
from flashinfer.utils import get_compute_capability

N = 3072
K_L = 768
FP8 = torch.float8_e4m3fn
# On SM90 the FP8 tensor-core (WGMMA) accumulation leaves an error floor that
# scales with Q[m, n] = sqrt(sum_r sum_k (a_scale_r[m] a_r[m, k] w_scale_r[n]
# w_r[n, k]) ** 2): the elementwise bound adds SM90_FLOOR_C1 * Q there. SM100 /
# SM103 accumulate in fp32 and keep the bound without it.
SM90_FLOOR_C1 = 2**-10
# (M, N, K_l) calls that select the lever programs a route may deliver, one call
# per program (``_lever_cases``). The two calls above 32768 rows reach the
# programs that the drainer lever leaves to larger calls at 4 and 8 ranks.
LEVER_SHAPES = (
    (1, 1024, 128),
    (1, 1024, 384),
    (16384, 12288, 4096),
    (32769, 1024, 128),
    (49152, 4096, 4096),
)


def _visible_arch():
    """The delivered architecture of ``cuda:0`` (``None`` without a GPU or outside the delivery)."""

    if torch.cuda.device_count() == 0:
        return None
    capability = tuple(get_compute_capability(torch.device("cuda:0")))
    arch = loader.CAPABILITY_ARCH.get(capability)
    return arch if arch in loader.ROUTES else None


def _world_size_delivered():
    arch = _visible_arch()
    return arch is not None and torch.cuda.device_count() in loader.COMM_MODES[arch]


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
    """fp32 all-reduce of the scaled partial products.

    Returns (bf16 result, sum |P_r|, Q): Q is the SM90 floor scale (None elsewhere).
    """

    partial = (a.float() @ w.float().t()) * a_scale[:, None] * w_scale[None, :]
    total = partial.clone()
    dist.all_reduce(total, op=dist.ReduceOp.SUM, group=group)
    magnitude = partial.abs()
    dist.all_reduce(magnitude, op=dist.ReduceOp.SUM, group=group)
    floor_scale = _sm90_floor_scale(a, a_scale, w, w_scale, group)
    return total.to(torch.bfloat16), magnitude, floor_scale


def _sm90_floor_scale(a, a_scale, w, w_scale, group):
    """SM90 floor scale Q = sqrt(sum_r sum_k (sa * a * sw * w) ** 2), else None."""

    if _visible_arch() != "sm_90a":
        return None
    square = a.float().square() @ w.float().square().t()
    square *= a_scale.square()[:, None] * w_scale.square()[None, :]
    dist.all_reduce(square, op=dist.ReduceOp.SUM, group=group)
    return square.sqrt()


def _check(actual, expected):
    """|y - r| <= 1e-2 + 1e-2 * sum_r |P_r| (+ SM90_FLOOR_C1 * Q on SM90)
    elementwise and relative L2 within 2**-7."""

    reference, magnitude, floor_scale = expected
    assert actual.dtype == torch.bfloat16 and actual.shape == reference.shape
    error = (actual.float() - reference.float()).abs()
    bound = 1e-2 + 1e-2 * magnitude
    if floor_scale is not None:
        bound = bound + SM90_FLOOR_C1 * floor_scale
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


def _lever_cases(world_size, arch, sm_count):
    """``(capacity, route key, program, call)`` of every lever program a route of
    ``world_size`` delivers on ``arch``. The call is the first ``LEVER_SHAPES``
    call that the loader dispatches to that program, or ``None``: as in
    ``call_program``, the capacity class gives the route and the program variant,
    and the lever of the call is taken in that variant.
    """

    capacities = [None] + [
        int(capacity)
        for variant_world_size, capacity, _suffix in loader.CAPACITY_VARIANTS.get(
            arch, []
        )
        if int(variant_world_size) == world_size
    ]
    cases = []
    for capacity in capacities:
        base = loader.route_key(world_size, arch, capacity or 1024)
        for lever in loader.lever_suffixes(arch)[1:]:
            program = loader.ROUTES[arch].get(base + lever)
            if program is None or program == loader.ROUTES[arch][base]:
                continue
            shapes = [
                (m, n, k_l)
                for m, n, k_l in LEVER_SHAPES
                if (capacity is None or m <= capacity)
                and loader.route_key(world_size, arch, capacity or m) == base
                and loader.lever_variant(
                    world_size,
                    m,
                    n,
                    k_l,
                    arch,
                    sm_count=sm_count,
                    variant=loader.program_variant(world_size, capacity or m, arch),
                )
                == lever
            ]
            cases.append((capacity, base + lever, program, (shapes or [None])[0]))
    return cases


def _run_cake_group(rank: int, world_size: int, port: int, symm_backend: str):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    if symm_backend == "NVSHMEM":
        symm_mem.set_backend("NVSHMEM")
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    group = dist.new_group(ranks=list(range(world_size)), backend="nccl")
    symm_mem.enable_symm_mem_for_group(group.group_name)
    generator = torch.Generator(device=device)
    generator.manual_seed(1188 + rank)
    arch = loader.device_facts(device.index).arch
    assert arch is not None and arch in loader.ROUTES
    comm_mode = loader.comm_mode(world_size, arch)

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
    assert workspace.rows_capacity == 1024
    assert workspace.s_bytes == loader.workspace_bytes(1024, N, world_size, arch)
    assert (workspace.arch, workspace.comm_mode) == (arch, comm_mode)
    assert workspace.program == loader.main_program(world_size, arch, 1024)
    assert workspace.captured is False
    if loader.GEOMETRY[arch]["GRID_RULE"] == loader.GRID_RULE_CLUSTER_PAIRS:
        assert workspace.max_clusters > 0 and workspace.sm_count == 0
    else:
        assert workspace.sm_count > 0 and workspace.max_clusters == 0
    if comm_mode == "nvls":
        # The nvls carrier binds the receive workspace through its multicast
        # address; the backend refuses to run without one.
        assert workspace.multicast is not None and workspace.multicast_ptr > 0
        assert workspace.multicast.dtype == torch.uint32
        assert workspace.multicast.data_ptr() == workspace.multicast_ptr
    else:
        assert workspace.multicast is None and workspace.multicast_ptr == 0
    if rank == 0:
        print(
            "[cake fp8 gemm+allreduce e2e] "
            f"arch={arch} world={world_size} carrier={comm_mode} symm={symm_backend} "
            f"clusters={workspace.max_clusters} sms={workspace.sm_count} "
            f"flags_in_pad={workspace.flags_in_signal_pad} "
            f"multicast={workspace.multicast_ptr:#x}",
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
    assert launcher.workspace_bytes == loader.workspace_bytes(4133, N, world_size, arch)
    assert launcher.program == loader.main_program(world_size, arch, 4133)
    assert launcher.max_active_clusters == workspace.max_clusters
    # Every rank must prepare with the same capacity: a differing max_rows is
    # refused on every rank by the prepare-time agreement, before any allocation.
    with pytest.raises(ValueError, match="max_rows must be identical"):
        prepare_fp8_rowwise_gemm_allreduce(*sample, group, max_rows=4133 + rank)
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

    # Launchers prepared from here on stay alive until the lever programs ran:
    # torch's NVSHMEM backend (before pytorch#192579) may return the cached,
    # smaller handle of a freed symmetric buffer at the same address.
    held = []
    # A capacity that selects a per-capacity program variant (when the
    # architecture delivers one for this world size) is served by that program.
    for variant_world_size, capacity, suffix in loader.CAPACITY_VARIANTS.get(arch, []):
        if int(variant_world_size) != world_size:
            continue
        variant_sample = _inputs(1, N, K_L, device, generator)
        variant_launcher = prepare_fp8_rowwise_gemm_allreduce(
            *variant_sample, group, max_rows=int(capacity)
        )
        assert variant_launcher.program.endswith("") and (
            variant_launcher.program
            == loader.main_program(world_size, arch, int(capacity))
        )
        assert loader.route_key(world_size, arch, int(capacity)).endswith(suffix)
        for served_rows in (1, int(capacity) - 1, int(capacity)):
            served = _inputs(served_rows, N, K_L, device, generator)
            _check(variant_launcher(*served), _expected(*served, group))
            del served
        held.append(variant_launcher)
        del variant_launcher, variant_sample

    # Every lever program a route delivers (the loader's LEVER_DISPATCH) computes
    # the route's result: one prepared launcher per delivered lever key, called
    # with a shape that selects that program.
    sm_count = loader.device_facts(device.index).sm_count
    for capacity, key, program, shape in _lever_cases(world_size, arch, sm_count):
        assert shape is not None, f"no e2e shape selects {key} on {arch}"
        m, n, k_l = shape
        lever_sample = _inputs(m, n, k_l, device, generator)
        lever_launcher = prepare_fp8_rowwise_gemm_allreduce(
            *lever_sample, group, max_rows=capacity or m
        )
        assert program in lever_launcher.programs
        assert (
            loader.call_program(
                world_size, arch, capacity or m, m, n, k_l, sm_count=sm_count
            )
            == program
        )
        _check(lever_launcher(*lever_sample), _expected(*lever_sample, group))
        held.append(lever_launcher)
        del lever_launcher, lever_sample
    del held

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
    _visible_arch() is None,
    reason="Cake fused FP8 GEMM + all-reduce e2e requires GPUs of a delivered architecture",
)
@pytest.mark.skipif(
    not _world_size_delivered(),
    reason="Cake fused FP8 GEMM + all-reduce e2e requires a delivered world size of visible GPUs",
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
def test_fp8_rowwise_gemm_allreduce_cake_visible_ranks(symm_backend):
    world_size = torch.cuda.device_count()
    port = random.randint(30000, 60000)
    mp.spawn(
        _run_cake_group,
        args=(world_size, port, symm_backend),
        nprocs=world_size,
        join=True,
    )
