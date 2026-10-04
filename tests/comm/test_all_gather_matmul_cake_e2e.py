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

Multi-GPU numerical test of the Cake all-gather matmul backend on an NCCL
subgroup of two, four or eight SM100 / SM103 devices.
"""

import gc
import random
import weakref

import pytest
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.multiprocessing as mp

from flashinfer.comm import all_gather_matmul, prepare_all_gather_matmul
from flashinfer.utils import get_compute_capability

PACKED_QKV_N_BY_WORLD_SIZE = {4: 2560, 8: 1280}


def _expected(inp, weight, group, world_size):
    gathered = torch.empty(
        world_size * inp.shape[0], inp.shape[1], dtype=inp.dtype, device=inp.device
    )
    dist.all_gather_into_tensor(gathered, inp, group=group)
    return (gathered.float() @ weight.float()).to(inp.dtype)


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
    inp = torch.randn(rows, 8192, dtype=dtype, device=device)
    weight = torch.randn(8192, 2048, dtype=dtype, device=device)
    expected = _expected(inp, weight, group, world_size)

    # Two consecutive calls return fresh outputs and the first is not overwritten.
    first = all_gather_matmul(inp, weight, group, backend="cake")
    torch.testing.assert_close(first, expected, atol=1e-2, rtol=1e-2)
    first_snapshot = first.clone()
    second = all_gather_matmul(inp, weight, group, backend="cake")
    assert first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(first, first_snapshot, atol=0, rtol=0)
    torch.testing.assert_close(second, expected, atol=1e-2, rtol=1e-2)

    # The backend callable writes a caller-provided output in place; a strided
    # input is rejected, not copied.
    from flashinfer.comm.all_gather_matmul.cake_all_gather_matmul import (
        all_gather_matmul_cake,
    )

    out = torch.full(
        (world_size * rows, 2048), float("nan"), dtype=dtype, device=device
    )
    assert all_gather_matmul_cake(inp, weight, group, backend="cake", out=out) is out
    torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)
    with pytest.raises(ValueError, match="contiguous"):
        all_gather_matmul(inp.t().contiguous().t(), weight, group, backend="cake")

    # The backend keeps no reference to the caller's tensors.
    inp_ref = weakref.ref(inp)
    weight_ref = weakref.ref(weight)
    torch.cuda.synchronize(device)
    del first, first_snapshot, second, expected, out, inp, weight
    gc.collect()
    assert inp_ref() is None
    assert weight_ref() is None

    # A different input on another stream reuses the same symmetric scratch.
    producer_stream = torch.cuda.Stream(device=device)
    torch.manual_seed(141 + rank)
    with torch.cuda.stream(producer_stream):
        inp = torch.randn(rows, 8192, dtype=dtype, device=device)
        weight = torch.randn(8192, 2048, dtype=dtype, device=device)
        expected = _expected(inp, weight, group, world_size)
        result = all_gather_matmul(inp, weight, group, backend="cake")
    torch.cuda.current_stream(device).wait_stream(producer_stream)
    torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)
    del inp, weight, expected, result

    packed_n = PACKED_QKV_N_BY_WORLD_SIZE.get(world_size)
    capability = get_compute_capability(device)
    packed_routed = packed_n is not None and (world_size == 8 or capability == (10, 3))
    if packed_routed and dtype == torch.bfloat16:
        packed_rows = 512
        packed_weight = torch.randn(8192, packed_n, dtype=dtype, device=device)
        active_inp = torch.randn(packed_rows, 8192, dtype=dtype, device=device)
        packed_expected = _expected(active_inp, packed_weight, group, world_size)
        launcher = prepare_all_gather_matmul(
            active_inp, packed_weight, group, backend="cake"
        )
        packed_stream = torch.cuda.Stream(device=device)
        packed_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(packed_stream):
            packed_first = launcher(active_inp)
            packed_first_snapshot = packed_first.clone()
            active_inp.neg_()
            packed_second = launcher(active_inp)
        torch.cuda.current_stream(device).wait_stream(packed_stream)
        assert packed_first.data_ptr() != packed_second.data_ptr()
        torch.testing.assert_close(packed_first, packed_first_snapshot, atol=0, rtol=0)
        torch.testing.assert_close(packed_first, packed_expected, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(
            packed_second, -packed_expected, atol=1e-2, rtol=1e-2
        )
        with pytest.raises(ValueError, match="shape"):
            launcher(torch.randn(packed_rows * 2, 8192, dtype=dtype, device=device))

    torch.cuda.synchronize(device)
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
