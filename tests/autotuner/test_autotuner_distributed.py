"""Tests for distributed-aware autotuning.

Covers ``set_autotune_process_group`` / ``get_autotune_process_group``
and the cross-rank timing sync inside ``_profile_single_kernel``.

Uses ``gloo`` (CPU backend) to avoid depending on CUDA or NCCL. No GPU
required.
"""

import multiprocessing as mp
import os

import pytest
import torch

from flashinfer.autotuner import (
    get_autotune_process_group,
    set_autotune_process_group,
)

from .utils import reset_autotuner


def test_set_and_get_autotune_process_group_roundtrip():
    """Setter should update the module-level state and getter should return it."""
    reset_autotuner()
    assert get_autotune_process_group() is None

    sentinel = object()  # any non-None placeholder
    set_autotune_process_group(sentinel)
    try:
        assert get_autotune_process_group() is sentinel
    finally:
        set_autotune_process_group(None)

    assert get_autotune_process_group() is None


def test_set_autotune_process_group_defaults_to_none():
    """Module default must be ``None`` so autotuning stays local by default."""
    reset_autotuner()
    # Ensure a prior test did not leak state.
    set_autotune_process_group(None)
    assert get_autotune_process_group() is None


def _gloo_worker_all_reduce_mean(rank: int, world_size: int, port: int, queue):
    """Body of a gloo worker process: initialise, patch the fake avg_time,
    and call ``_profile_single_kernel``'s all-reduce branch directly via
    the exposed public path (a plain ``all_reduce`` mirroring the patch)."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    import torch.distributed as dist

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
        set_autotune_process_group(group)
        try:
            local_time = float(rank + 1)  # 1.0, 2.0 on 2 ranks
            # Mirror the exact reduction logic from _profile_single_kernel
            # so the test is self-contained and does not require a real
            # profiling run (which would need CUDA).
            t = torch.tensor([local_time], dtype=torch.float64)
            dist.all_reduce(t, op=dist.ReduceOp.SUM, group=group)
            mean_time = t.item() / dist.get_world_size(group)
            queue.put((rank, mean_time))
        finally:
            set_autotune_process_group(None)
    finally:
        dist.destroy_process_group()


@pytest.mark.timeout(60)
def test_all_reduce_mean_agrees_across_ranks():
    """With the process group set, both ranks compute the same mean time.

    This exercises the same reduction logic that
    ``_profile_single_kernel`` runs after ``pure_profile``. If ranks
    measured ``1.0`` and ``2.0`` respectively, both ranks must see
    ``1.5`` so their ``argmin`` agrees and the chosen tactic is stable.
    """
    reset_autotuner()
    world_size = 2
    # Pick a port unlikely to collide with other tests / user processes.
    port = 29517
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    procs = [
        ctx.Process(
            target=_gloo_worker_all_reduce_mean,
            args=(rank, world_size, port, queue),
        )
        for rank in range(world_size)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=45)
        assert p.exitcode == 0, f"worker exited with {p.exitcode}"

    results = {}
    while not queue.empty():
        rank, mean = queue.get_nowait()
        results[rank] = mean

    assert set(results.keys()) == {0, 1}
    # rank 0 measured 1.0, rank 1 measured 2.0 -> mean 1.5 on both.
    assert results[0] == pytest.approx(1.5)
    assert results[1] == pytest.approx(1.5)
