"""One-shot all-reduce on a workspace whose Lamport sub-buffer exceeds 1 GiB.

The Lamport workspace holds three sub-buffers of ``comm_size`` bytes; each call
writes one and clears the one two calls back. Above 1 GiB the byte offset of the
third sub-buffer (``2 * comm_size``) no longer fits in a 32-bit int.
"""

import multiprocessing as mp
import socket

import pytest
import torch
import torch.distributed as dist

import flashinfer.comm as comm
from flashinfer.comm.mnnvl import TorchDistBackend

WORLD_SIZE = 2
HIDDEN = 8192
# comm_size = world_size * max_token_num * hidden * 2 bytes = 1.04 GiB.
MAX_TOKEN_NUM = 34000
TOKENS = 16
# Enough calls for every sub-buffer to be reused after a clear at offset 2.
NUM_CALLS = 9
# Lamport sub-buffers (3 * comm_size) plus the two-shot buffer (comm_size).
REQUIRED_BYTES_PER_RANK = 4 * WORLD_SIZE * MAX_TOKEN_NUM * HIDDEN * 2


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _worker(rank: int, port: int) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=WORLD_SIZE,
    )
    group = dist.group.WORLD
    workspace = None
    try:
        workspace = comm.create_allreduce_fusion_workspace(
            backend="trtllm",
            world_size=WORLD_SIZE,
            rank=rank,
            max_token_num=MAX_TOKEN_NUM,
            hidden_dim=HIDDEN,
            dtype=torch.bfloat16,
            comm_backend=TorchDistBackend(group=group),
        )
        for call in range(NUM_CALLS):
            torch.manual_seed(100 * call + rank)
            x = torch.randn(TOKENS, HIDDEN, device=device, dtype=torch.bfloat16)
            ref = x.float().clone()
            dist.all_reduce(ref, group=group)
            out = torch.empty_like(x)
            comm.allreduce_fusion(
                input=x,
                workspace=workspace,
                pattern=comm.AllReduceFusionPattern.kAllReduce,
                output=out,
                use_oneshot=True,
            )
            torch.cuda.synchronize()
            torch.testing.assert_close(
                out.float(),
                ref,
                atol=5e-2,
                rtol=1e-2,
                msg=lambda m: f"call {call}: {m}",
            )
    finally:
        if workspace is not None:
            workspace.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


def test_oneshot_allreduce_comm_size_above_1gib() -> None:
    if torch.cuda.device_count() < WORLD_SIZE:
        pytest.skip(f"requires at least {WORLD_SIZE} GPUs")
    for rank in range(WORLD_SIZE):
        free, _ = torch.cuda.mem_get_info(rank)
        if free < REQUIRED_BYTES_PER_RANK + (2 << 30):
            pytest.skip(
                f"requires {REQUIRED_BYTES_PER_RANK / 2**30:.1f} GiB free on cuda:{rank}"
            )
    port = _get_open_port()
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_worker, args=(rank, port)) for rank in range(WORLD_SIZE)
    ]
    for proc in procs:
        proc.start()
    for rank, proc in enumerate(procs):
        proc.join()
        assert proc.exitcode == 0, f"rank {rank} failed with exit code {proc.exitcode}"
