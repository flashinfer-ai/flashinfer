"""One-shot Lamport kernels on a workspace whose sub-buffer exceeds 1 GiB.

The Lamport workspace holds three sub-buffers of ``comm_size`` bytes; each call
writes one and clears the one two calls back. Above 1 GiB the byte offset of the
third sub-buffer (``2 * comm_size``) no longer fits in a 32-bit int. Every call
gets new inputs, so a sub-buffer left with the data of an earlier call shows.
"""

import multiprocessing as mp
import socket
import sys
import time

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
# Exit code of a rank whose TCPStore port was taken between _get_open_port and its own bind.
PORT_IN_USE_EXIT_CODE = 98
MAX_PORT_ATTEMPTS = 3


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _init_process_group(rank: int, port: int, backend: str = "nccl") -> None:
    try:
        dist.init_process_group(
            backend=backend,
            init_method=f"tcp://localhost:{port}",
            rank=rank,
            world_size=WORLD_SIZE,
        )
    except RuntimeError as error:
        if "EADDRINUSE" in str(error) or "address already in use" in str(error):
            sys.exit(PORT_IN_USE_EXIT_CODE)
        raise


def _run_ranks(worker) -> None:
    """Run one process per rank; on a taken port, start over on a new one."""
    ctx = mp.get_context("spawn")
    for _ in range(MAX_PORT_ATTEMPTS):
        port = _get_open_port()
        procs = [
            ctx.Process(target=worker, args=(rank, port)) for rank in range(WORLD_SIZE)
        ]
        for proc in procs:
            proc.start()
        # A rank that exits early leaves its peers in the unbounded Lamport
        # wait (or waiting on the store), so stop them instead of joining.
        while any(p.is_alive() for p in procs) and all(
            p.exitcode in (None, 0) for p in procs
        ):
            time.sleep(0.2)
        failed = any(p.exitcode not in (None, 0) for p in procs)
        port_in_use = any(p.exitcode == PORT_IN_USE_EXIT_CODE for p in procs)
        for proc in procs:
            if failed and proc.is_alive():
                proc.kill()
            proc.join()
        if not port_in_use:
            break
    exit_codes = [proc.exitcode for proc in procs]
    assert all(code == 0 for code in exit_codes), f"rank exit codes: {exit_codes}"


def _worker(rank: int, port: int) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    _init_process_group(rank, port)
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


def _moe_finalize_worker(rank: int, port: int) -> None:
    """The MoE finalize all-reduce, which has its own copy of the Lamport helper."""
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    _init_process_group(rank, port)
    group = dist.group.WORLD
    ipc_handles = None
    try:
        ipc_handles, workspace_tensor = (
            comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
                rank, WORLD_SIZE, MAX_TOKEN_NUM, HIDDEN, group=group
            )
        )
        dtype, top_k = torch.bfloat16, 8
        for call in range(NUM_CALLS):
            # The same inputs on every rank, new ones on every call.
            gen = torch.Generator().manual_seed(call)
            shared = torch.randn(TOKENS, HIDDEN, generator=gen).to(device, dtype)
            fc2 = torch.randn(TOKENS * top_k, HIDDEN, generator=gen).to(device, dtype)
            scale = torch.randn(TOKENS, top_k, generator=gen).to(device, dtype)
            idx = torch.randint(
                0, TOKENS * top_k, (TOKENS, top_k), generator=gen, dtype=torch.int32
            ).to(device)
            residual = torch.randn(TOKENS, HIDDEN, generator=gen).to(device, dtype)
            norm_weight = torch.randn(HIDDEN, generator=gen).to(device, dtype)
            norm_out = torch.empty_like(residual)
            residual_out = torch.empty_like(residual)
            quant_out = torch.empty(TOKENS * HIDDEN // 4, dtype=dtype, device=device)
            scale_out = torch.empty(
                comm.compute_fp4_swizzled_layout_sf_size(TOKENS, HIDDEN // 16),
                dtype=torch.uint8,
                device=device,
            )
            comm.trtllm_moe_finalize_allreduce_fusion(
                allreduce_in=fc2,
                residual_in=residual,
                norm_weight=norm_weight,
                expanded_idx_to_permuted_idx=idx,
                workspace_ptrs=workspace_tensor,
                launch_with_pdl=False,
                world_rank=rank,
                world_size=WORLD_SIZE,
                eps=1e-5,
                shared_expert_output=shared,
                expert_scale_factor=scale,
                norm_out=norm_out,
                residual_out=residual_out,
                quant_out=quant_out,
                scale_out=scale_out,
                routed_scaling_factor=1.0,
            )
            torch.cuda.synchronize()
            per_rank = (
                torch.sum(fc2[idx].float() * scale.unsqueeze(-1).float(), dim=1)
                + shared.float()
            ).to(dtype)
            ref = (per_rank * WORLD_SIZE + residual).float()
            torch.testing.assert_close(
                residual_out.float(),
                ref,
                atol=0.2,
                rtol=0.2,
                msg=lambda m: f"call {call}: {m}",
            )
    finally:
        dist.barrier(group=group)
        if ipc_handles is not None:
            comm.trtllm_destroy_ipc_workspace_for_all_reduce_fusion(
                ipc_handles, group=group
            )
        dist.destroy_process_group(group=group)


@pytest.mark.parametrize(
    "worker",
    [
        pytest.param(_worker, id="allreduce"),
        pytest.param(_moe_finalize_worker, id="moe_finalize"),
    ],
)
def test_oneshot_comm_size_above_1gib(worker) -> None:
    if torch.cuda.device_count() < WORLD_SIZE:
        pytest.skip(f"requires at least {WORLD_SIZE} GPUs")
    for rank in range(WORLD_SIZE):
        free, _ = torch.cuda.mem_get_info(rank)
        if free < REQUIRED_BYTES_PER_RANK + (2 << 30):
            pytest.skip(
                f"requires {REQUIRED_BYTES_PER_RANK / 2**30:.1f} GiB free on cuda:{rank}"
            )
    _run_ranks(worker)
