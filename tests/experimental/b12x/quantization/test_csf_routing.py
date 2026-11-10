"""Presence masks preserve sparse routes across frozen CUDA graph replay."""

import pytest
import torch

from b12x._lib.quant.csf_routing import compile_csf_active_experts, mark_active_experts
from b12x._lib.runtime_control import kernel_resolution_guard
from ..conftest import require_b12x


@pytest.mark.parametrize("dtype", (torch.int32, torch.int64))
@pytest.mark.parametrize("experts", (288, 512))
def test_presence_mask_invalid_duplicates_empty_and_graph(dtype, experts):
    device = require_b12x()
    invalid = 2**32 + 3 if dtype == torch.int64 else experts
    ids = torch.tensor(
        ([3, -1, 3, 19, experts - 1, invalid] * 97), dtype=dtype, device=device
    )
    active = torch.empty(experts, dtype=torch.int32, device=device)
    program = compile_csf_active_experts(dtype == torch.int64)
    with kernel_resolution_guard("CSF presence mask"):
        for count in (0, 1, 8, 63, 256, len(ids)):
            active.fill_(123)
            mark_active_experts(ids[:count], active, program)
            expected = torch.zeros_like(active)
            valid = ids[:count][(ids[:count] >= 0) & (ids[:count] < experts)].long()
            expected[valid] = 1
            assert torch.equal(active, expected)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            mark_active_experts(ids, active, program)
        for value in (0, -1, experts - 1, invalid):
            ids.fill_(value)
            active.fill_(123)
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            expected = torch.zeros_like(active)
            if 0 <= value < experts:
                expected[value] = 1
            assert torch.equal(active, expected)
        graph.reset()
