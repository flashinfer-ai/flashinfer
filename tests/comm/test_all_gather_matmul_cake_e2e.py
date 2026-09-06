import gc
import importlib
import random
import weakref

import pytest
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.multiprocessing as mp

from flashinfer.comm import all_gather_matmul, prepare_all_gather_matmul
from flashinfer.utils import get_compute_capability


def _run_cake_subgroup(rank: int, world_size: int, port: int, dtype: torch.dtype):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
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
    torch.manual_seed(41 + rank)
    rows = 384
    inp = symm_mem.empty(rows, 8192, dtype=dtype, device=device).normal_()
    weight = torch.randn(8192, 2048, dtype=dtype, device=device)

    gathered = torch.empty(world_size * rows, 8192, dtype=dtype, device=device)
    dist.all_gather_into_tensor(gathered, inp, group=group)
    expected = gathered @ weight

    if (
        world_size == 8
        and dtype == torch.bfloat16
        and get_compute_capability(device) == (10, 3)
    ):
        packed_weight = torch.randn(8192, 1280, dtype=dtype, device=device)
        active_inp = symm_mem.empty(rows, 8192, dtype=dtype, device=device).normal_()
        packed_gathered = torch.empty(
            world_size * rows, 8192, dtype=dtype, device=device
        )
        dist.all_gather_into_tensor(packed_gathered, active_inp, group=group)
        packed_expected = packed_gathered @ packed_weight
        packed_launcher = prepare_all_gather_matmul(
            inp, packed_weight, group, backend="cake"
        )
        packed_stream = torch.cuda.Stream(device=device)
        packed_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(packed_stream):
            packed_first = packed_launcher(active_inp)
            packed_first_snapshot = packed_first.clone()
            packed_second = packed_launcher(active_inp)
        torch.cuda.current_stream(device).wait_stream(packed_stream)
        assert packed_first.data_ptr() != packed_second.data_ptr()
        torch.testing.assert_close(packed_first, packed_first_snapshot, atol=0, rtol=0)
        torch.testing.assert_close(packed_first, packed_expected, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(packed_second, packed_expected, atol=1e-2, rtol=1e-2)
        del (
            active_inp,
            packed_expected,
            packed_first,
            packed_first_snapshot,
            packed_gathered,
            packed_launcher,
            packed_second,
            packed_stream,
            packed_weight,
        )

    first = all_gather_matmul(inp, weight, group, backend="cake")
    torch.testing.assert_close(first, expected, atol=1e-2, rtol=1e-2)
    first_snapshot = first.clone()
    second = all_gather_matmul(inp, weight, group, backend="cake")
    assert first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(first, first_snapshot, atol=0, rtol=0)
    torch.testing.assert_close(second, expected, atol=1e-2, rtol=1e-2)

    backend = importlib.import_module(
        "flashinfer.comm.all_gather_matmul.cake_all_gather_matmul"
    )
    assert (
        sum(
            len(workspace.descriptor_cache)
            for workspace in backend._WORKSPACES.values()
        )
        == 1
    )
    inp_ref = weakref.ref(inp)
    weight_ref = weakref.ref(weight)
    torch.cuda.synchronize(device)
    del first, first_snapshot, second, expected, gathered, inp, weight
    gc.collect()
    assert inp_ref() is None
    assert weight_ref() is None

    for workspace in backend._WORKSPACES.values():
        workspace.descriptor_cache.clear()
    backend._DESCRIPTOR_CACHE_MAX_ENTRIES = 1

    producer_stream = torch.cuda.Stream(device=device)
    owner_stream = torch.cuda.Stream(device=device)
    cached_use_stream = torch.cuda.Stream(device=device)
    torch.manual_seed(141 + rank)
    with torch.cuda.stream(producer_stream):
        inp = torch.randn(rows, 8192, dtype=dtype, device=device)
        weight = torch.randn(8192, 2048, dtype=dtype, device=device)
        gathered = torch.empty(world_size * rows, 8192, dtype=dtype, device=device)
        dist.all_gather_into_tensor(gathered, inp, group=group)
        expected = gathered @ weight
    owner_stream.wait_stream(producer_stream)
    cold_miss_gate = torch.cuda.Event(enable_timing=False)
    with torch.cuda.stream(owner_stream):
        torch.cuda._sleep(2_000_000_000)
        cold_miss_gate.record()
        ordinary = all_gather_matmul(inp, weight, group, backend="cake")
    assert not cold_miss_gate.query()
    pinned_descriptor_churn = [
        torch.empty(384, dtype=torch.uint8, device="cpu", pin_memory=True).fill_(index)
        for index in range(32)
    ]
    torch.cuda.current_stream(device).wait_stream(owner_stream)
    torch.testing.assert_close(ordinary, expected, atol=1e-2, rtol=1e-2)

    descriptor_caches = [
        workspace.descriptor_cache for workspace in backend._WORKSPACES.values()
    ]
    assert sum(len(cache) for cache in descriptor_caches) == 1
    descriptor_cache = next(cache for cache in descriptor_caches if cache)
    descriptor_ptr = next(iter(descriptor_cache.values())).data_ptr()

    cached_use_stream.wait_stream(producer_stream)
    cached_use_complete = torch.cuda.Event(enable_timing=False)
    with torch.cuda.stream(cached_use_stream):
        torch.cuda._sleep(2_000_000_000)
        cached = all_gather_matmul(inp, weight, group, backend="cake")
        cached_use_complete.record()
    assert not cached_use_complete.query()

    input_ptr = inp.data_ptr()
    weight_ptr = weight.data_ptr()
    inp_ref = weakref.ref(inp)
    weight_ref = weakref.ref(weight)
    del inp, weight, gathered, ordinary
    assert inp_ref() is None
    assert weight_ref() is None

    with torch.cuda.stream(producer_stream):
        replacement_inp = torch.empty(rows, 8192, dtype=dtype, device=device).normal_()
        replacement_weight = torch.empty(
            8192, 2048, dtype=dtype, device=device
        ).normal_()
        replacement_gathered = torch.empty(
            world_size * rows, 8192, dtype=dtype, device=device
        )
        dist.all_gather_into_tensor(replacement_gathered, replacement_inp, group=group)
        replacement_expected = replacement_gathered @ replacement_weight
    assert replacement_inp.data_ptr() != input_ptr
    assert replacement_weight.data_ptr() != weight_ptr

    eviction_stream = torch.cuda.Stream(device=device)
    eviction_stream.wait_stream(producer_stream)
    with torch.cuda.stream(eviction_stream):
        replacement = all_gather_matmul(
            replacement_inp, replacement_weight, group, backend="cake"
        )
    assert sum(len(cache) for cache in descriptor_caches) == 1
    assert all(
        descriptor.data_ptr() != descriptor_ptr
        for cache in descriptor_caches
        for descriptor in cache.values()
    )
    assert not cached_use_complete.query()

    with torch.cuda.stream(owner_stream):
        descriptor_churn = [
            torch.empty(384, dtype=torch.uint8, device=device) for _ in range(32)
        ]
    assert descriptor_ptr not in {tensor.data_ptr() for tensor in descriptor_churn}

    torch.cuda.current_stream(device).wait_stream(cached_use_stream)
    torch.cuda.current_stream(device).wait_stream(producer_stream)
    torch.cuda.current_stream(device).wait_stream(owner_stream)
    torch.cuda.current_stream(device).wait_stream(eviction_stream)
    torch.testing.assert_close(cached, expected, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(replacement, replacement_expected, atol=1e-2, rtol=1e-2)
    del (
        cached,
        expected,
        replacement,
        replacement_expected,
        replacement_gathered,
        replacement_inp,
        replacement_weight,
        descriptor_churn,
        pinned_descriptor_churn,
    )
    gc.collect()

    dist.destroy_process_group(group)
    dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() not in (2, 4, 8),
    reason="Cake all-gather matmul e2e requires exactly two, four, or eight visible GPUs",
)
@pytest.mark.skipif(
    torch.cuda.device_count() == 0
    or get_compute_capability(torch.device("cuda:0")) not in ((10, 0), (10, 3)),
    reason="Cake all-gather matmul e2e requires SM100 or SM103",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_all_gather_matmul_cake_arbitrary_subgroup(dtype):
    world_size = torch.cuda.device_count()
    port = random.randint(30000, 60000)
    mp.spawn(
        _run_cake_subgroup,
        args=(world_size, port, dtype),
        nprocs=world_size,
        join=True,
    )
