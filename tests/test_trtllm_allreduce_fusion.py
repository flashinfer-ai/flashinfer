import multiprocessing as mp
import socket
from typing import Any

import numpy as np
import pytest
import torch
import torch.distributed as dist

import flashinfer.comm as comm

# todo(Yingyi): add benchmark and quant test

# Usage: test var
kOneShotMaxTokenNum = 128
MIN_TOKEN_NUM = 1
MAX_TOKEN_NUM = 2048
SF_VEC_SIZE = 16

# temp var
SCALE_FACTOR_RANGE = (-1, 1)


def _run_correctness_worker(world_size, rank, dtype, hidden_dim, distributed_init_port):
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
        token_nums = [1, 128, 1024, 2048]
        pattern_codes = [
            comm.AllReduceFusionPattern.kAllReduce,
            comm.AllReduceFusionPattern.kARResidualRMSNorm,
            comm.AllReduceFusionPattern.kARResidualRMSNormFP8Quant,
            comm.AllReduceFusionPattern.kARResidualRMSNormFP4Quant,
            comm.AllReduceFusionPattern.kARResidualRMSNormOutFP8Quant,
            comm.AllReduceFusionPattern.kARResidualRMSNormOutFP4Quant,
        ]
        swizzled_layout_codes = [
            comm.QuantizationSFLayout.LINEAR,
            comm.QuantizationSFLayout.SWIZZLED_128x4,
            comm.QuantizationSFLayout.SWIZZLED_8x4,
        ]
        launch_with_pdls = [True, False]
        use_oneshots = [True, False, None]
        trigger_completion_at_ends = [True, False]
        fp32_accs = [True, False]

        lamport_use_fp32 = dtype == torch.float32

        # create workspace for allreduce fusion
        ipc_handles, workspace_tensor = (
            comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
                rank,
                world_size,
                MAX_TOKEN_NUM,
                hidden_dim,
                group=group,
                use_fp32_lamport=lamport_use_fp32,
            )
        )

        test_loop = 5

        for token_num in token_nums:
            for pattern_code in pattern_codes:
                for swizzled_layout_code in swizzled_layout_codes:
                    for launch_with_pdl in launch_with_pdls:
                        for use_oneshot in use_oneshots:
                            for trigger_completion_at_end in trigger_completion_at_ends:
                                for fp32_acc in fp32_accs:
                                    if token_num < world_size and not use_oneshot:
                                        continue
                                    if dtype == torch.float32 and (
                                        pattern_code
                                        == comm.AllReduceFusionPattern.kARResidualRMSNormOutFP4Quant
                                        or pattern_code
                                        == comm.AllReduceFusionPattern.kARResidualRMSNormFP4Quant
                                    ):
                                        continue

                                    dist.barrier(group=group)
                                    test_passed = True
                                    print(
                                        f"test RANK {rank}: token{token_num}-hidden_dim{hidden_dim}-dtype{dtype}-pattern{pattern_code}-layout{swizzled_layout_code}-pdl{launch_with_pdl} start"
                                    )
                                    dist.barrier(group=group)
                                    torch.cuda.synchronize()

                                    message_size = token_num * hidden_dim

                                    allreduce_in = torch.randn(
                                        message_size, dtype=dtype, device=device
                                    )
                                    allreduce_in_clone = allreduce_in.clone()

                                    all_reduce_out = torch.zeros(
                                        message_size, dtype=dtype, device=device
                                    )

                                    residual_in = torch.randn(
                                        message_size, dtype=dtype, device=device
                                    )
                                    residual_in_clone = residual_in.clone()

                                    residual_out = torch.empty_like(residual_in)
                                    norm_out = torch.empty_like(residual_in)
                                    quant_out = torch.empty(
                                        message_size, dtype=dtype, device=device
                                    )

                                    scale_out = None
                                    assert hidden_dim % SF_VEC_SIZE == 0, (
                                        "hidden_dim must be divisible by SF_VEC_SIZE"
                                    )
                                    if (
                                        swizzled_layout_code
                                        == comm.QuantizationSFLayout.SWIZZLED_128x4
                                    ):
                                        # TODO(Yingyi): check this
                                        padded_message_size = (
                                            (token_num + 127) // 128 * 128
                                        ) * ((hidden_dim + 63) // 64 * 4)
                                        scale_out = torch.empty(
                                            padded_message_size,
                                            dtype=dtype,
                                            device=device,
                                        )
                                    else:
                                        scale_out = torch.empty(
                                            message_size // SF_VEC_SIZE,
                                            dtype=dtype,
                                            device=device,
                                        )

                                    rms_gamma = torch.randn(
                                        hidden_dim, dtype=dtype, device=device
                                    )
                                    scale_factor = (
                                        torch.rand(
                                            1, dtype=torch.float32, device=device
                                        )
                                        * (
                                            SCALE_FACTOR_RANGE[1]
                                            - SCALE_FACTOR_RANGE[0]
                                        )
                                        + SCALE_FACTOR_RANGE[0]
                                    )
                                    rms_eps = 1e-3

                                    # warmup
                                    s = torch.cuda.Stream()
                                    s.wait_stream(torch.cuda.current_stream())
                                    with torch.cuda.stream(s):
                                        for _ in range(test_loop):
                                            comm.trtllm_allreduce_fusion(
                                                allreduce_in=allreduce_in,
                                                world_size=world_size,
                                                world_rank=rank,
                                                token_num=token_num,
                                                hidden_dim=hidden_dim,
                                                workspace_ptrs=workspace_tensor,
                                                launch_with_pdl=launch_with_pdl,
                                                use_oneshot=use_oneshot,
                                                trigger_completion_at_end=trigger_completion_at_end,
                                                fp32_acc=fp32_acc,
                                                pattern_code=pattern_code,
                                                allreduce_out=all_reduce_out,
                                                residual_in=residual_in,
                                                residual_out=residual_out,
                                                norm_out=norm_out,
                                                quant_out=quant_out,
                                                scale_out=scale_out,
                                                rms_gamma=rms_gamma,
                                                rms_eps=rms_eps,
                                                scale_factor=scale_factor,
                                                layout_code=swizzled_layout_code,
                                            )

                                    # NOTE: in real case, you dont have to set all optional params. You could set those required by fusion pattern.
                                    # capture
                                    g = torch.cuda.CUDAGraph()
                                    with torch.cuda.graph(g):
                                        for _ in range(test_loop):
                                            comm.trtllm_allreduce_fusion(
                                                allreduce_in=allreduce_in,
                                                world_size=world_size,
                                                world_rank=rank,
                                                token_num=token_num,
                                                hidden_dim=hidden_dim,
                                                workspace_ptrs=workspace_tensor,
                                                launch_with_pdl=launch_with_pdl,
                                                use_oneshot=use_oneshot,
                                                trigger_completion_at_end=trigger_completion_at_end,
                                                fp32_acc=fp32_acc,
                                                pattern_code=pattern_code,
                                                allreduce_out=all_reduce_out,
                                                residual_in=residual_in,
                                                residual_out=residual_out,
                                                norm_out=norm_out,
                                                quant_out=quant_out,
                                                scale_out=scale_out,
                                                rms_gamma=rms_gamma,
                                                rms_eps=rms_eps,
                                                scale_factor=scale_factor,
                                                layout_code=swizzled_layout_code,
                                            )
                                    # replay
                                    g.replay()
                                    torch.cuda.synchronize()

                                    # match shape
                                    all_reduce_out = all_reduce_out.view(
                                        token_num, hidden_dim
                                    )
                                    residual_out = residual_out.view(
                                        token_num, hidden_dim
                                    )
                                    norm_out = norm_out.view(token_num, hidden_dim)

                                    torch.cuda.synchronize()

                                    # calculate reference
                                    # allreduce_out
                                    dist.all_reduce(allreduce_in_clone, group=group)
                                    ref_allreduce_out = allreduce_in_clone.clone()
                                    ref_allreduce_out = ref_allreduce_out.view(
                                        token_num, hidden_dim
                                    ).to(torch.float32)

                                    # residual_out
                                    ref_residual_out = (
                                        ref_allreduce_out
                                        + residual_in_clone.view(
                                            token_num, hidden_dim
                                        ).to(torch.float32)
                                    )

                                    # norm_out
                                    variance = (
                                        ref_residual_out.to(torch.float32)
                                        .pow(2)
                                        .mean(dim=-1, keepdim=True)
                                    )
                                    hidden_states = ref_residual_out * torch.rsqrt(
                                        variance + rms_eps
                                    )
                                    ref_norm_out = (
                                        rms_gamma.to(torch.float32) * hidden_states
                                    )

                                    # check correctness
                                    tolerance = 8e-2 if dtype == torch.float16 else 8e-1
                                    # compare allreduce_out
                                    if (
                                        pattern_code
                                        == comm.AllReduceFusionPattern.kAllReduce
                                    ):
                                        torch.testing.assert_close(
                                            all_reduce_out.to(torch.float32),
                                            ref_allreduce_out,
                                            atol=tolerance,
                                            rtol=1e-2,
                                        )
                                    elif (
                                        pattern_code
                                        == comm.AllReduceFusionPattern.kARResidualRMSNormOutFP8Quant
                                        or pattern_code
                                        == comm.AllReduceFusionPattern.kARResidualRMSNormOutFP4Quant
                                    ):
                                        torch.testing.assert_close(
                                            residual_out.to(torch.float32),
                                            ref_residual_out,
                                            atol=tolerance,
                                            rtol=1e-2,
                                        )

                                        torch.testing.assert_close(
                                            norm_out.to(torch.float32),
                                            ref_norm_out,
                                            atol=tolerance,
                                            rtol=1e-2,
                                        )

                                        # todo(Yingyi): check quant out
                                    dist.barrier(group=group)
                                    if test_passed:
                                        print(
                                            f"test RANK {rank}: token{token_num}-hidden_dim{hidden_dim}-dtype{dtype}-pattern{pattern_code}-layout{swizzled_layout_code}-pdl{launch_with_pdl} passed"
                                        )
                                    else:
                                        print(
                                            f"test RANK {rank}: token{token_num}-hidden_dim{hidden_dim}-dtype{dtype}-pattern{pattern_code}-layout{swizzled_layout_code}-pdl{launch_with_pdl} failed"
                                        )
    finally:
        dist.barrier(group=group)

        comm.trtllm_destroy_ipc_workspace_for_all_reduce(ipc_handles, group=group)

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
    world_size: int,
    dtype: torch.dtype,
    hidden_dim: int,
    test_target: Any,
    target_args: tuple = (),
) -> None:
    mp.set_start_method("spawn", force=True)

    procs = []
    distributed_init_port = get_open_port()
    for i in range(world_size):
        proc_args = (
            world_size,
            i,
            dtype,
            hidden_dim,
            distributed_init_port,
        ) + target_args
        proc = mp.Process(target=test_target, args=proc_args, name=f"Worker-{i}")
        proc.start()
        procs.append(proc)

    for i in range(world_size):
        procs[i].join()
        assert procs[i].exitcode == 0, (
            f"Process {i} failed with exit code {procs[i].exitcode}"
        )


@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hidden_dim", [1024, 2048, 4096, 7168, 8192])
def test_trtllm_allreduce_fusion(world_size, dtype, hidden_dim):
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    available_gpus = torch.cuda.device_count()
    if world_size > available_gpus:
        raise ValueError(
            f"world_size {world_size} is greater than available_gpus {available_gpus}"
        )
    print(f"Running test for world_size={world_size}")

    multi_process_parallel(
        world_size,
        dtype,
        hidden_dim,
        _run_correctness_worker,
        target_args=(),
    )
    print(f"allreduce fusion tp = {world_size}: OK")
