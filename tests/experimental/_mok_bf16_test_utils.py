# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Shared process-group lifetime for the MoK distributed tests."""

import datetime
import os

import pytest


@pytest.fixture(scope="session")
def mok_distributed_group():
    if "RANK" not in os.environ or int(os.environ.get("WORLD_SIZE", "0")) not in (
        1,
        4,
        16,
        64,
    ):
        pytest.skip(
            "Launch MoK distributed tests with torchrun using 1, 4, 16 or 64 ranks"
        )
    import torch
    import torch.distributed as dist

    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(
        device
    ) not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("Requires an SM100a-compatible CUDA device")
    torch.cuda.set_device(device)
    owns_group = not dist.is_initialized()
    if owns_group:
        # A rank that JIT-compiles the kernels on a cold cache can hold the other
        # ranks in the workspace all-gather for minutes (longer on slow shared
        # filesystems); a too-short collective timeout then aborts the process
        # group from the NCCL watchdog.  MOK_PG_TIMEOUT_S widens it for such runs.
        timeout_s = int(os.environ.get("MOK_PG_TIMEOUT_S", "600"))
        dist.init_process_group(
            "nccl", device_id=device, timeout=datetime.timedelta(seconds=timeout_s)
        )
    try:
        yield dist.group.WORLD
    finally:
        if owns_group:
            torch.cuda.synchronize()
            dist.barrier()
            dist.destroy_process_group()
