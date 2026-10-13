# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Opt-in four/eight-rank transport tests; ordinary pytest runs skip GPU collectives."""

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist


def pytest_addoption(parser):
    group = parser.getgroup("rdma-moe-alltoall")
    group.addoption(
        "--rdma-moe-dtype", choices=("bf16", "fp8", "nvfp4"), default="nvfp4"
    )
    group.addoption(
        "--rdma-moe-hidden", type=int, choices=(4096, 6144, 7168), default=4096
    )
    group.addoption(
        "--rdma-moe-mode", choices=("auto", "legacy", "coalesced"), default="auto"
    )
    group.addoption("--rdma-moe-full", action="store_true")


def pytest_generate_tests(metafunc):
    if "transport_config" not in metafunc.fixturenames:
        return
    if metafunc.config.getoption("--rdma-moe-full"):
        values = [
            (dtype, hidden)
            for dtype in ("bf16", "fp8", "nvfp4")
            for hidden in (4096, 6144, 7168)
        ]
    else:
        values = [
            (
                metafunc.config.getoption("--rdma-moe-dtype"),
                metafunc.config.getoption("--rdma-moe-hidden"),
            )
        ]
    metafunc.parametrize(
        "transport_config",
        values,
        ids=[f"{dtype}-h{hidden}" for dtype, hidden in values],
    )


@pytest.fixture(scope="session")
def distributed_device():
    if int(os.environ.get("WORLD_SIZE", "1")) not in (4, 8):
        pytest.skip(
            "Launch with torchrun --nproc-per-node=4 or 8 to test GPU collectives"
        )
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if torch.cuda.get_device_capability(device) != (12, 0):
        pytest.skip("The validated low-latency transport target is SM120")
    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group("nccl", timeout=timedelta(minutes=5), device_id=device)
    yield device
    torch.cuda.synchronize(device)
    if owns_group:
        dist.destroy_process_group()


@pytest.fixture
def transport_factory(request, distributed_device):
    from flashinfer.comm import RdmaMoeAlltoAll, RdmaMoeAlltoAllConfig

    mode = request.config.getoption("--rdma-moe-mode")

    def create(dtype, hidden, capacity=8):
        return RdmaMoeAlltoAll(
            dist.group.WORLD,
            capacity,
            hidden,
            288,
            config=RdmaMoeAlltoAllConfig(
                communication_dtype=dtype, communication_mode=mode
            ),
        )

    return create


@pytest.fixture
def transport_buffer(transport_factory, transport_config):
    buffer = transport_factory(*transport_config)
    try:
        yield buffer
    finally:
        torch.cuda.synchronize()
        buffer.close()


@pytest.fixture(params=("legacy", "coalesced"))
def planned_mode(request):
    return request.param
