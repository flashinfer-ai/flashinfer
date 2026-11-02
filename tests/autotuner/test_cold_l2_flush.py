# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The cold-L2 flush must evict by reading, so it leaves no dirty lines."""

import importlib

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

implementation = importlib.import_module("flashinfer.autotuner.autotuner")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_bytes", [4096, 4099])
def test_flush_buffer_is_zeroed_float32(num_bytes):
    buf = implementation._make_l2_flush_buffer(num_bytes, torch.device("cuda"))
    assert buf.dtype == torch.float32
    assert buf.numel() * buf.element_size() >= num_bytes
    assert torch.count_nonzero(buf).item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flush_reads_without_writing_the_buffer():
    buf = implementation._make_l2_flush_buffer(8 << 20, torch.device("cuda"))
    # A known non-zero pattern: a flush that wrote the buffer (zero_() did)
    # would change it, whatever kernel it used.
    buf.fill_(1.5)
    implementation._flush_l2(buf)  # warm up the reduction kernel
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        implementation._flush_l2(buf)
        torch.cuda.synchronize()
    torch.testing.assert_close(buf, torch.full_like(buf, 1.5), rtol=0, atol=0)
    kernels = [
        e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
    ]
    # The read is a reduction (plus the memset of its scalar output); a fill
    # (zero_()) or a cast copy (an integer sum) would be another kernel over
    # the buffer, whatever that kernel is called.
    names = [k.lower() for k in kernels]
    assert names, "no CUDA kernel was profiled for the flush"
    assert any("reduce" in k for k in names), kernels
    assert all("reduce" in k or "memset" in k for k in names), kernels


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flush_after_allocation_does_not_allocate_device_memory():
    # In MoE EP tuning the flush runs right after a distributed barrier; a
    # device allocation there could fail on one rank only. The helper's own
    # flush leaves the reduction's scratch in the caching allocator, so a flush
    # on a freshly made buffer takes no new segment, even from an empty cache.
    # Same stream as the helper's flush: the allocator pools blocks per stream.
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    buf = implementation._make_l2_flush_buffer(8 << 20, torch.device("cuda"))
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()["segment.all.allocated"]
    implementation._flush_l2(buf)
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["segment.all.allocated"] == before
