# SPDX-License-Identifier: Apache-2.0
"""Two CPU control-plane participants; not a multi-GPU kernel test."""

from datetime import timedelta

import torch.distributed as dist
import torch.multiprocessing as mp


def _reject_worker(rank, init_method):
    from flashinfer.experimental.ulysses.sm100 import prepare

    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=init_method,
        timeout=timedelta(seconds=30),
    )
    try:
        # Both ranks reject before importing or allocating the kernel. Their
        # failures intentionally differ; neither may strand the other rank.
        try:
            prepare(
                group=dist.group.WORLD,
                local_seq=18944 if rank == 0 else 17,
                used_seqlen=37807,
                full_nvlink=False,
                discard_tail_output=True,
            )
        except RuntimeError as exc:
            assert "rank 0" in str(exc) and "rank 1" in str(exc)
            assert "before allocation" in str(exc)
        else:
            raise AssertionError("invalid requests must reject collectively")
    finally:
        dist.destroy_process_group()


def test_collective_rejection_before_gpu_resources(tmp_path):
    mp.spawn(
        _reject_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True
    )
