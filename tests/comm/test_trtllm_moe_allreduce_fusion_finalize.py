import multiprocessing as mp
import os
import socket
import time
from typing import Any

import numpy as np
import pytest
import torch
import torch.distributed as dist

import flashinfer.comm as comm

# todo(Yingyi): add benchmark and quant test

# Usage: test var
kOneShotMaxTokenNum = 128
MAX_TOKEN_NUM = 2048
HIDDEN_SIZE = 7168
MAX_EXPERT_NUM = 16
SF_VEC_SIZE = 16

# temp var
SCALE_FACTOR_RANGE = (-1, 1)


def _run_correctness_worker(
    world_size,
    rank,
    dtype,
    distributed_init_port,
    shared_expert_output,
    fc2_output,
    scale,
    expanded_idx_to_permuted_idx,
    residual,
):
    def rms_norm(x: torch.Tensor, weight: torch.Tensor = None, eps: float = 1e-6):
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        if weight is not None:
            y = y * weight
        return y

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    distributed_init_method = f"tcp://localhost:{distributed_init_port}"
    dist.init_process_group(
        backend="nccl",
        init_method=distributed_init_method,
        rank=rank,
        world_size=world_size,
    )
    group = dist.group.WORLD

    try:
        device = torch.device(f"cuda:{rank}")
        seq_lens = [16]
        top_k = 8
        eps = 1e-5

        launch_with_pdls = [True, False]

        # create workspace for moe allreduce fusion
        ipc_handles, workspace_tensor = (
            comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
                rank, world_size, MAX_TOKEN_NUM, HIDDEN_SIZE, group=group
            )
        )

        test_loop = 5

        for seq_len in seq_lens:
            for launch_with_pdl in launch_with_pdls:
                dist.barrier(group=group)
                test_passed = True
                print(
                    f"test RANK {rank}: seq_len{seq_len}-topk{top_k}-tp{world_size}-{dtype}-pdl{launch_with_pdl} start"
                )
                dist.barrier(group=group)
                torch.cuda.synchronize()
                for _ in range(test_loop):
                    # == Generate input ==
                    # move to local device
                    shared_expert_output = shared_expert_output.to(device)
                    fc2_output = fc2_output.to(device)
                    scale = scale.to(device)
                    expanded_idx_to_permuted_idx = expanded_idx_to_permuted_idx.to(
                        device
                    )
                    residual = residual.to(device)

                    # make clone
                    fc2_output_clone = fc2_output.clone()
                    norm_weight = torch.randn(
                        (HIDDEN_SIZE,), dtype=dtype, device=device
                    )

                    norm_out = torch.empty_like(residual)
                    residual_out = torch.empty_like(residual)
                    seq_len, hidden_size = norm_out.shape
                    quant_out = torch.empty(
                        seq_len * hidden_size // 4, dtype=dtype, device=device
                    )
                    scale_out = torch.empty(
                        seq_len * hidden_size // SF_VEC_SIZE, dtype=dtype, device=device
                    )
                    routed_scaling_factor = 2.5

                    # == Run kernel ==
                    torch.cuda.synchronize()
                    s = torch.cuda.Stream()
                    s.wait_stream(torch.cuda.current_stream())
                    # warmup
                    with torch.cuda.stream(s):
                        for _ in range(test_loop):
                            comm.trtllm_moe_finalize_allreduce_fusion(
                                allreduce_in=fc2_output,
                                residual_in=residual,
                                norm_weight=norm_weight,
                                expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                                workspace_ptrs=workspace_tensor,
                                launch_with_pdl=launch_with_pdl,
                                world_rank=rank,
                                world_size=world_size,
                                eps=eps,
                                shared_expert_output=shared_expert_output,
                                expert_scale_factor=scale,
                                norm_out=norm_out,
                                residual_out=residual_out,
                                quant_out=quant_out,
                                scale_out=scale_out,
                                routed_scaling_factor=routed_scaling_factor,
                            )
                    torch.cuda.current_stream().wait_stream(s)

                    # capture
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        for _ in range(test_loop):
                            comm.trtllm_moe_finalize_allreduce_fusion(
                                allreduce_in=fc2_output,
                                residual_in=residual,
                                norm_weight=norm_weight,
                                expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                                workspace_ptrs=workspace_tensor,
                                launch_with_pdl=launch_with_pdl,
                                world_rank=rank,
                                world_size=world_size,
                                eps=eps,
                                shared_expert_output=shared_expert_output,
                                expert_scale_factor=scale,
                                norm_out=norm_out,
                                residual_out=residual_out,
                                quant_out=quant_out,
                                scale_out=scale_out,
                                routed_scaling_factor=routed_scaling_factor,
                            )

                    # replay
                    g.replay()

                    torch.cuda.synchronize()

                    # == Calculate reference output ==
                    expert_reduction = torch.sum(
                        fc2_output_clone[expanded_idx_to_permuted_idx]
                        * scale.unsqueeze(-1)
                        * routed_scaling_factor,
                        dim=1,
                    )

                    torch_before_residual = (
                        expert_reduction + shared_expert_output
                    ) * world_size
                    torch_residual = torch_before_residual + residual
                    torch_residual = torch_residual.to(torch.float32)
                    torch_output_hidden_states = rms_norm(
                        torch_residual, norm_weight, eps
                    ).to(dtype)

                    # == Check correctness ==
                    if not torch.allclose(
                        residual_out.to(torch.float32),
                        torch_residual.to(torch.float32),
                        rtol=0.2,
                        atol=0.2,
                    ):
                        test_passed = False
                        print(f"Rank {rank} residual_out mismatch")
                        print(f"residual_out: {residual_out}")
                        print(f"torch_residual: {torch_residual}")
                        print(
                            f"max diff: {torch.max(torch.abs(residual_out.to(torch.float32) - torch_residual.to(torch.float32)))}"
                        )
                        print(
                            f"max diff idx: {torch.argmax(torch.abs(residual_out.to(torch.float32) - torch_residual.to(torch.float32)))}"
                        )
                        print(
                            f"max diff value: {residual_out.to(torch.float32).view(-1)[torch.argmax(torch.abs(residual_out.to(torch.float32) - torch_residual.to(torch.float32)))]}"
                        )
                        print(
                            f"max diff ref value: {torch_residual.to(torch.float32).view(-1)[torch.argmax(torch.abs(residual_out.to(torch.float32) - torch_residual.to(torch.float32)))]}"
                        )

                    if not torch.allclose(
                        norm_out.to(torch.float32),
                        torch_output_hidden_states.to(torch.float32),
                        rtol=0.2,
                        atol=0.2,
                    ):
                        test_passed = False
                        print(f"Rank {rank} norm_out mismatch")
                        print(f"norm_out: {norm_out}")
                        print(
                            f"torch_output_hidden_states: {torch_output_hidden_states}"
                        )
                        print(
                            f"max diff: {torch.max(torch.abs(norm_out.to(torch.float32) - torch_output_hidden_states.to(torch.float32)))}"
                        )
                        print(
                            f"max diff idx: {torch.argmax(torch.abs(norm_out.to(torch.float32) - torch_output_hidden_states.to(torch.float32)))}"
                        )
                        print(
                            f"max diff value: {norm_out.to(torch.float32).view(-1)[torch.argmax(torch.abs(norm_out.to(torch.float32) - torch_output_hidden_states.to(torch.float32)))]}"
                        )
                        print(
                            f"max diff ref value: {torch_output_hidden_states.to(torch.float32).view(-1)[torch.argmax(torch.abs(norm_out.to(torch.float32) - torch_output_hidden_states.to(torch.float32)))]}"
                        )

                    torch.testing.assert_close(
                        residual_out.to(torch.float32),
                        torch_residual.to(torch.float32),
                        rtol=0.2,
                        atol=0.2,
                    )
                    torch.testing.assert_close(
                        norm_out.to(torch.float32),
                        torch_output_hidden_states.to(torch.float32),
                        rtol=0.2,
                        atol=0.2,
                    )

                dist.barrier(group=group)
                if test_passed:
                    print(
                        f"test RANK {rank}: seq_len{seq_len}-topk{top_k}-tp{world_size}-{dtype}-pdl{launch_with_pdl} passed"
                    )
                else:
                    print(
                        f"test RANK {rank}: seq_len{seq_len}-topk{top_k}-tp{world_size}-{dtype}-pdl{launch_with_pdl} failed"
                    )
    finally:
        dist.barrier(group=group)

        comm.trtllm_destroy_ipc_workspace_for_all_reduce_fusion(
            ipc_handles, group=group
        )

        dist.destroy_process_group(group=group)


def _run_back_to_back_worker(world_size, rank, dtype, distributed_init_port):
    """Replays many back-to-back launches, so a rank can start the next launch
    while its peers are still in the previous one. Exits with code 2 if the
    launches do not finish."""
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{distributed_init_port}",
        rank=rank,
        world_size=world_size,
    )
    group = dist.group.WORLD
    token_num, top_k = kOneShotMaxTokenNum, 8
    _, workspace_tensor = comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
        rank, world_size, MAX_TOKEN_NUM, HIDDEN_SIZE, group=group
    )
    torch.manual_seed(0)
    fc2_output = torch.randn(token_num * top_k, HIDDEN_SIZE, dtype=dtype, device=device)
    expanded_idx_to_permuted_idx = torch.arange(
        token_num * top_k, dtype=torch.int32, device=device
    ).view(token_num, top_k)
    scale = torch.rand(token_num, top_k, dtype=dtype, device=device)
    residual = torch.randn(token_num, HIDDEN_SIZE, dtype=dtype, device=device)
    norm_weight = torch.ones(HIDDEN_SIZE, dtype=dtype, device=device)
    norm_out = torch.empty_like(residual)
    residual_out = torch.empty_like(residual)

    def run():
        comm.trtllm_moe_finalize_allreduce_fusion(
            allreduce_in=fc2_output,
            residual_in=residual,
            norm_weight=norm_weight,
            expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
            workspace_ptrs=workspace_tensor,
            launch_with_pdl=True,
            world_rank=rank,
            world_size=world_size,
            eps=1e-5,
            shared_expert_output=None,
            expert_scale_factor=scale,
            norm_out=norm_out,
            residual_out=residual_out,
            quant_out=None,
            scale_out=None,
            routed_scaling_factor=1.0,
        )

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        run()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(20):
            run()
    torch.cuda.synchronize()
    dist.barrier(group=group)
    for _ in range(50):
        g.replay()
    done = torch.cuda.Event()
    done.record()
    deadline = time.monotonic() + 60
    while not done.query():
        if time.monotonic() > deadline:
            print(f"Rank {rank}: back-to-back launches did not finish", flush=True)
            os._exit(2)
        time.sleep(0.01)

    # Every rank has the same inputs.
    expected = residual.float() + world_size * (
        fc2_output.float()[expanded_idx_to_permuted_idx.long()]
        * scale.float().unsqueeze(-1)
    ).sum(dim=1)
    torch.testing.assert_close(residual_out.float(), expected, rtol=2e-2, atol=0.5)
    dist.barrier(group=group)
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
    world_size: int, dtype: torch.dtype, test_target: Any, target_args: tuple = ()
) -> None:
    mp.set_start_method("spawn", force=True)

    procs = []
    distributed_init_port = get_open_port()
    for i in range(world_size):
        proc_args = (world_size, i, dtype, distributed_init_port) + target_args
        proc = mp.Process(target=test_target, args=proc_args, name=f"Worker-{i}")
        proc.start()
        procs.append(proc)

    for i in range(world_size):
        procs[i].join()
        assert procs[i].exitcode == 0, (
            f"Process {i} failed with exit code {procs[i].exitcode}"
        )


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_trtllm_moe_finalize_allreduce_fusion(world_size, dtype):
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    available_gpus = torch.cuda.device_count()
    if world_size > available_gpus:
        pytest.skip(
            f"world_size {world_size} is greater than available_gpus {available_gpus}"
        )
    print(f"Running test for world_size={world_size}")

    # generate shared random input tensor across all ranks
    seq_len = 16
    hidden_size = 7168
    top_k = 8

    shared_expert_output = torch.randn((seq_len, hidden_size), dtype=dtype)
    fc2_output = torch.randn((seq_len * top_k, hidden_size), dtype=dtype)
    scale = torch.randn((seq_len, top_k), dtype=dtype)
    expanded_idx_to_permuted_idx = torch.randint(
        0, seq_len * top_k, (seq_len, top_k), dtype=torch.int32
    )
    residual = torch.randn_like(shared_expert_output)

    multi_process_parallel(
        world_size,
        dtype,
        _run_correctness_worker,
        target_args=(
            shared_expert_output,
            fc2_output,
            scale,
            expanded_idx_to_permuted_idx,
            residual,
        ),
    )
    print(f"moe finalize allreduce fusion tp = {world_size}: OK")


@pytest.mark.parametrize("world_size", [2, 4])
def test_trtllm_moe_finalize_allreduce_fusion_back_to_back(world_size):
    if world_size > torch.cuda.device_count():
        pytest.skip(f"world_size {world_size} is greater than available GPUs")
    multi_process_parallel(world_size, torch.bfloat16, _run_back_to_back_worker)


if __name__ == "__main__":
    test_trtllm_moe_finalize_allreduce_fusion(2, torch.float16)
