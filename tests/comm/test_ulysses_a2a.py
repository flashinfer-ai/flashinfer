# flashinfer: fused-transpose Ulysses NVLink-P2P all-to-all correctness test.
# The kernel under test is adapted from ThunderKittens' NVLink all-to-all:
# https://github.com/HazyResearch/ThunderKittens/blob/main/kernels/parallel/all_to_all/all_to_all.cu

import multiprocessing as mp
import socket
from typing import Any

import pytest
import torch
import torch.distributed as dist

import flashinfer.comm as comm

# head_dim==2 Ulysses layout: [B, S_local, H, D] <-> [B, S_global, H_local, D]
# (B, S_local, H, D) configs; H must be divisible by every tested world size.
TEST_SHAPES = [
    (1, 8, 8, 128),  # vec-aligned fast path
    (2, 16, 24, 64),  # vec-aligned, batch > 1
    (1, 5, 8, 3),  # unaligned row -> scalar fallback path
]
DTYPES = [torch.float16, torch.bfloat16, torch.float32]


def _initialize_process_group(world_size, rank, distributed_init_port):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{distributed_init_port}",
        rank=rank,
        world_size=world_size,
    )
    return dist.group.WORLD


def _ref_input_a2a(x_local, world_size, rank, H_local, group):
    """All-gather-based reference for mode 0 (independent of the kernel).

    out_r[b, j*S_local + s, hl, d] = x_j[b, s, r*H_local + hl, d]
    """
    gathered = [torch.empty_like(x_local) for _ in range(world_size)]
    dist.all_gather(gathered, x_local.contiguous(), group=group)
    slabs = [xj[:, :, rank * H_local : (rank + 1) * H_local, :] for xj in gathered]
    return torch.cat(slabs, dim=1).contiguous()


def _run_correctness_worker(world_size, rank, distributed_init_port):
    group = _initialize_process_group(world_size, rank, distributed_init_port)
    device = torch.device(f"cuda:{rank}")
    custom_ptr = None
    buffer_ptrs = None
    meta_ptrs = None
    try:
        # Output staging buffer must hold the largest a2a output (== input numel).
        max_bytes = max(B * S * H * D for (B, S, H, D) in TEST_SHAPES) * 4  # fp32
        buffer_ptrs = comm.create_shared_buffer(max_bytes, group=group)
        # Signal buffers: one Signal per rank (same layout as vLLM custom AR).
        meta_ptrs = comm.create_shared_buffer(comm.vllm_meta_size(), group=group)

        custom_ptr = comm.init_ulysses_a2a(
            buffer_ptrs, meta_ptrs, rank, world_size, True
        )
        # init zeroed each rank's own signal; make it globally visible.
        dist.barrier(group=group)

        for dtype in DTYPES:
            for B, S_local, H, D in TEST_SHAPES:
                if H % world_size != 0:
                    continue
                H_local = H // world_size
                S_global = S_local * world_size

                torch.manual_seed(1234 + rank)  # distinct data per rank
                x = torch.randn(B, S_local, H, D, dtype=dtype, device=device)

                # ---- mode 0: [B,S_local,H,D] -> [B,S_global,H_local,D] ----
                ref = _ref_input_a2a(x, world_size, rank, H_local, group)
                out = torch.empty(B, S_global, H_local, D, dtype=dtype, device=device)
                comm.ulysses_a2a(custom_ptr, x, out, B, S_local, H, D, 0)
                torch.cuda.synchronize()
                assert torch.equal(out, ref), (
                    f"input a2a mismatch ws={world_size} rank={rank} "
                    f"dtype={dtype} shape=({B},{S_local},{H},{D})"
                )

                # ---- mode 1 round-trip: a2a(out, mode1) must recover x ----
                back = torch.empty(B, S_local, H, D, dtype=dtype, device=device)
                comm.ulysses_a2a(custom_ptr, out, back, B, S_local, H, D, 1)
                torch.cuda.synchronize()
                assert torch.equal(back, x), (
                    f"round-trip mismatch ws={world_size} rank={rank} "
                    f"dtype={dtype} shape=({B},{S_local},{H},{D})"
                )
        dist.barrier(group=group)
    finally:
        if custom_ptr is not None:
            comm.dispose_ulysses_a2a(custom_ptr)
        if buffer_ptrs:
            comm.free_shared_buffer(buffer_ptrs, group)
        if meta_ptrs:
            comm.free_shared_buffer(meta_ptrs, group)
        dist.destroy_process_group(group=group)


def get_open_port() -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]
    except OSError:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("::1", 0))
            return s.getsockname()[1]


def multi_process_parallel(
    world_size: int, test_target: Any, target_args: tuple = ()
) -> None:
    mp.set_start_method("spawn", force=True)
    procs = []
    port = get_open_port()
    for i in range(world_size):
        proc = mp.Process(
            target=test_target,
            args=(world_size, i, port) + target_args,
            name=f"Worker-{i}",
        )
        proc.start()
        procs.append(proc)
    for i in range(world_size):
        procs[i].join()
        assert procs[i].exitcode == 0, (
            f"Process {i} failed with exit code {procs[i].exitcode}"
        )


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_ulysses_a2a(world_size):
    available_gpus = torch.cuda.device_count()
    if world_size > available_gpus:
        pytest.skip(f"world_size {world_size} > available_gpus {available_gpus}")
    multi_process_parallel(world_size, _run_correctness_worker)
    print(f"ulysses a2a ws={world_size}: OK")
