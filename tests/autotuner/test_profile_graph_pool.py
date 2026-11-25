# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Profiling captures must not leave one private pool per profiled tactic."""

import pytest
import torch

from flashinfer.autotuner import TuningConfig, autotune

from .utils import DummyRunner, reset_autotuner


class AllocatingRunner(DummyRunner):
    """Allocates during capture, as ops with internal workspaces do."""

    def forward(self, inputs, tactic=-1, **kwargs):
        workspace = torch.empty(1 << 20, dtype=torch.uint8, device=inputs[0].device)
        workspace.fill_(tactic)
        return inputs[0]


def _private_pool_entries() -> int:
    stats = torch._C._cuda_memoryStats(torch.cuda.current_device())
    if "reserved_bytes_by_private_pools" not in stats:
        pytest.skip("this PyTorch does not report reserved_bytes_by_private_pools")
    return len(stats["reserved_bytes_by_private_pools"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "config",
    [
        TuningConfig(use_cuda_graph=True),
        TuningConfig(use_cuda_graph=True, use_cold_l2_cache=True),
        TuningConfig(use_cuda_graph=True, use_cuda_graph_individual_timing=True),
    ],
    ids=["hot", "cold_l2", "individual"],
)
def test_profiling_captures_share_one_pool(config):
    tuner = reset_autotuner()
    inputs = [torch.zeros(16, device="cuda")]
    runner = AllocatingRunner(valid_tactics=tuple(range(8)))
    before = _private_pool_entries()
    with autotune(tune_mode=True):
        _, tactic = tuner.choose_one(
            "profile_graph_pool_test", [runner], config, inputs
        )
    torch.cuda.synchronize()
    # A fallback tactic or a recorded failure means nothing was measured.
    assert tactic in range(8)
    assert "profile_graph_pool_test" not in tuner.stats.failed_profiling_count
    # Each capture without a shared pool adds one entry per profiled tactic.
    assert _private_pool_entries() - before <= 1
