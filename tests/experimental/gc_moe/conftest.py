# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for the experimental GC MoE tests."""

import os
import socket

import pytest
import torch

from .reference import BACKEND_BY_CAPABILITY


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="session")
def backend():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    capability = tuple(torch.cuda.get_device_capability())
    try:
        return BACKEND_BY_CAPABILITY[capability]
    except KeyError:
        pytest.skip(f"GC MoE has no backend for compute capability {capability}")


@pytest.fixture(scope="session")
def gc_group(backend):
    """One-rank expert-parallel group for the single-rank numerical references."""
    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if dist.is_initialized():
        if dist.get_world_size() > 1:
            pytest.skip(
                "the numerical tests compare one rank against a single-rank "
                "reference; launch without torchrun"
            )
        yield dist.group.WORLD
        return
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{_free_port()}",
        world_size=1,
        rank=0,
        device_id=torch.device("cuda", local_rank),
    )
    try:
        yield dist.group.WORLD
    finally:
        dist.destroy_process_group()


@pytest.fixture(scope="module")
def workspace(gc_group):
    import flashinfer

    from .reference import GEOMETRY

    with flashinfer.GCMoEWorkspace(
        gc_group,
        GEOMETRY["num_experts"],
        GEOMETRY["capacity"],
        GEOMETRY["num_topk"],
        GEOMETRY["hidden"],
        GEOMETRY["intermediate_hidden"],
    ) as created:
        yield created
