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
subgroup of two, four or eight SM100 / SM103 devices: both weight layouts,
tail row counts, scratch growth (including the stale rendezvous-handle guard)
and the capacity-bound prepared launcher.
"""

import gc
import random
import warnings
import weakref

import pytest
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.multiprocessing as mp

from flashinfer.comm import all_gather_matmul, prepare_all_gather_matmul
from flashinfer.utils import get_compute_capability

K = 8192
# Llama-3.1-70B column-parallel widths per tensor-parallel degree (qkv, gate_up).
ENGINE_WIDTHS = {2: (2048,), 4: (2560, 14336), 8: (1280, 7168)}


def _expected(inp, weight, group, world_size):
    gathered = torch.empty(
        world_size * inp.shape[0], inp.shape[1], dtype=inp.dtype, device=inp.device
    )
    dist.all_gather_into_tensor(gathered, inp, group=group)
    return (gathered.float() @ weight.float()).to(inp.dtype)


def _check(actual, expected):
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


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
    inp = torch.randn(rows, K, dtype=dtype, device=device)
    weight = torch.randn(K, 2048, dtype=dtype, device=device)
    expected = _expected(inp, weight, group, world_size)

    # Two consecutive calls return fresh outputs and the first is not overwritten.
    first = all_gather_matmul(inp, weight, group, backend="cake")
    _check(first, expected)
    first_snapshot = first.clone()
    second = all_gather_matmul(inp, weight, group, backend="cake")
    assert first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(first, first_snapshot, atol=0, rtol=0)
    _check(second, expected)

    # The backend callable writes a caller-provided output in place; a strided
    # input is rejected, not copied.
    from flashinfer.comm.all_gather_matmul.cake_all_gather_matmul import (
        all_gather_matmul_cake,
    )

    out = torch.full(
        (world_size * rows, 2048), float("nan"), dtype=dtype, device=device
    )
    assert all_gather_matmul_cake(inp, weight, group, backend="cake", out=out) is out
    _check(out, expected)
    with pytest.raises(ValueError, match="contiguous"):
        all_gather_matmul(inp.t().contiguous().t(), weight, group, backend="cake")

    # The engine's [N, K] parameter is consumed through its transposed view
    # (no copy); a weight with other strides is rejected.
    param = torch.randn(2048, K, dtype=dtype, device=device)
    expected_k_major = _expected(inp, param.t(), group, world_size)
    _check(all_gather_matmul(inp, param.t(), group, backend="cake"), expected_k_major)
    with pytest.raises(ValueError, match="strides"):
        all_gather_matmul(
            inp,
            torch.randn(K, 4096, dtype=dtype, device=device)[:, :2048],
            group,
            backend="cake",
        )

    # A foreign symmetric buffer freed right before a growth: torch's
    # NVSHMEM allocator before pytorch#192579 hands the next allocation at that
    # address the freed buffer's cached rendezvous handle, and the backend
    # re-allocates past the undersized handle. The buffer is sized between the
    # current scratch (512 rows per peer) and the next one (1152 rows), the
    # issue's pattern; whether the heap reuses the address depends on its
    # state, so the detections are reported, not asserted.
    from flashinfer.comm.all_gather_matmul import cake_all_gather_matmul as backend

    foreign = symm_mem.empty(world_size, 640, K, dtype=dtype, device=device)
    symm_mem.rendezvous(foreign, group=group.group_name)
    del foreign
    torch.cuda.synchronize(device)
    dist.barrier(group=group)
    retries_before_tail = backend._RENDEZVOUS_STATS["stale_retries"]

    # Tail row counts: the output has exactly world_size * M rows and the
    # scratch grows once when a larger M arrives.
    for tail_rows in (125, 1025):
        tail_inp = torch.randn(tail_rows, K, dtype=dtype, device=device)
        tail_out = all_gather_matmul(tail_inp, param.t(), group, backend="cake")
        assert tail_out.shape == (world_size * tail_rows, 2048)
        _check(tail_out, _expected(tail_inp, param.t(), group, world_size))
        del tail_inp, tail_out
    if rank == 0:
        print(
            "[cake all-gather matmul e2e] stale rendezvous handles re-allocated after "
            f"the foreign free: {backend._RENDEZVOUS_STATS['stale_retries'] - retries_before_tail}",
            flush=True,
        )

    # Deterministic stale-handle injection on a real growth (1152 -> 2048 rows
    # per peer through the prepared path): the first rendezvous of the new
    # scratch is reported with half its buffer size, the backend parks that
    # allocation and re-allocates, the parked allocation is released and the
    # process-wide warning fires on the first detection only.
    real_rendezvous = symm_mem.rendezvous
    injected = {}

    class _Undersized:
        def __init__(self, handle):
            self._handle = handle

        def __getattr__(self, name):
            return getattr(self._handle, name)

        @property
        def buffer_size(self):
            return int(self._handle.buffer_size) // 2

    def undersized_rendezvous(tensor, group=None):
        handle = real_rendezvous(tensor, group=group)
        if "parked" not in injected and tensor.numel() == world_size * 2048 * K:
            injected["parked"] = weakref.ref(tensor)
            return _Undersized(handle)
        return handle

    retries_before = backend._RENDEZVOUS_STATS["stale_retries"]
    grown_inp = torch.randn(2048, K, dtype=dtype, device=device)
    symm_mem.rendezvous = undersized_rendezvous
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            grown = prepare_all_gather_matmul(
                grown_inp, param.t(), group, backend="cake", max_rows=2048
            )
    finally:
        symm_mem.rendezvous = real_rendezvous
    assert "parked" in injected
    assert backend._RENDEZVOUS_STATS["stale_retries"] == retries_before + 1
    warned = any("pytorch#192579" in str(w.message) for w in caught)
    assert warned == (retries_before == 0)
    grown_out = grown(grown_inp)
    _check(grown_out, _expected(grown_inp, param.t(), group, world_size))
    torch.cuda.synchronize(device)
    gc.collect()
    assert injected["parked"]() is None
    del grown, grown_inp, grown_out

    # The backend keeps no reference to the caller's tensors.
    inp_ref = weakref.ref(inp)
    weight_ref = weakref.ref(weight)
    torch.cuda.synchronize(device)
    del first, first_snapshot, second, expected, expected_k_major, out, inp, weight
    gc.collect()
    assert inp_ref() is None
    assert weight_ref() is None

    # A different input on another stream reuses the same symmetric scratch.
    producer_stream = torch.cuda.Stream(device=device)
    torch.manual_seed(141 + rank)
    with torch.cuda.stream(producer_stream):
        inp = torch.randn(rows, K, dtype=dtype, device=device)
        weight = torch.randn(K, 2048, dtype=dtype, device=device)
        expected = _expected(inp, weight, group, world_size)
        result = all_gather_matmul(inp, weight, group, backend="cake")
    torch.cuda.current_stream(device).wait_stream(producer_stream)
    _check(result, expected)
    del inp, weight, expected, result, param

    # Capacity-bound prepared launcher on the engine widths of this
    # tensor-parallel degree with the engine's [N, K] parameters: one
    # collective at preparation, then any row count up to max_rows.
    if dtype == torch.bfloat16:
        for n in ENGINE_WIDTHS[world_size]:
            engine_param = torch.randn(n, K, dtype=dtype, device=device)
            sample = torch.randn(512, K, dtype=dtype, device=device)
            launcher = prepare_all_gather_matmul(
                sample, engine_param.t(), group, backend="cake", max_rows=2048
            )
            prepared_stream = torch.cuda.Stream(device=device)
            prepared_stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(prepared_stream):
                prepared_first = launcher(sample)
                prepared_first_snapshot = prepared_first.clone()
                sample.neg_()
                prepared_second = launcher(sample)
            torch.cuda.current_stream(device).wait_stream(prepared_stream)
            sample.neg_()
            prepared_expected = _expected(sample, engine_param.t(), group, world_size)
            assert prepared_first.data_ptr() != prepared_second.data_ptr()
            torch.testing.assert_close(
                prepared_first, prepared_first_snapshot, atol=0, rtol=0
            )
            _check(prepared_first, prepared_expected)
            _check(prepared_second, -prepared_expected)
            for served_rows in (125, 1025, 2048):
                served = torch.randn(served_rows, K, dtype=dtype, device=device)
                served_out = launcher(served)
                assert served_out.shape == (world_size * served_rows, n)
                _check(
                    served_out, _expected(served, engine_param.t(), group, world_size)
                )
                del served, served_out
            with pytest.raises(ValueError, match=r"\[1, 2048\]"):
                launcher(torch.randn(2049, K, dtype=dtype, device=device))
            with pytest.raises(ValueError, match="contiguous"):
                launcher(torch.randn(512, 2 * K, dtype=dtype, device=device)[:, :K])
            del launcher, engine_param, sample, prepared_first, prepared_second
            del prepared_first_snapshot, prepared_expected

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
