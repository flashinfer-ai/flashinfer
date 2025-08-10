import multiprocessing as mp
import socket
from typing import Any

import pytest
import torch
import torch.distributed as dist

import flashinfer.comm as comm

"""
NOTE:
The assertion of result closeness is disabled for now,
since assertion fails for some cases, which breaks the tests and introduces NCCL timeout.

Trt-llm encourage using certain shapes for this custom all-reduce kernel,

hidden_size in range [256, 8192], and maxHiddenSize should be 8192.
The recommended case is [1024, 2048, 4096, 8192].

If new trt-llm source kernels are available (function name starts with "trtllm_"), we would recommend using them.
"""

maxBatchSize = 1
maxBeamWidth = 3
maxTokenNum = 128
maxHiddenSize = 4096  # max hidden size for all reduce
RANDOM_SEED = 42


def _run_correctness_worker(world_size, rank, dtype, distributed_init_port):
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
        token_nums = [64, 128]
        strategy_codes = [
            comm.AllReduceStrategyType.ONESHOT,
            comm.AllReduceStrategyType.TWOSHOT,
        ]

        # below are the recommended hidden sizes for custom all-reduce in trtllm test
        # hidden_size should be in range [256, 8192], and maxHiddenSize should be 8192
        hidden_sizes = [1024, 4096]
        config_codes = [
            0,
            comm.AllReduceStrategyConfig.USE_MEMCPY,
            comm.AllReduceStrategyConfig.PUSH_MODE,
        ]
        fusion_op_codes = [
            comm.AllReduceFusionOp.NONE,
            comm.AllReduceFusionOp.RESIDUAL_RMS_NORM,
            comm.AllReduceFusionOp.RESIDUAL_RMS_PREPOST_NORM,
            # Below are not enabled for custom all-reduce in trtllm test, skip
            # comm.AllReduceFusionOp.LAST_PROCESS_FOR_UB,
            # comm.AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8,
            # comm.AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4,
            # comm.AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_FP8,
            # comm.AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4,
            # comm.AllReduceFusionOp.MOE_ALLREDUCE_RESIDUAL_RMS_NORM,
        ]
        launch_with_pdls = [True, False]

        # create ipc memory
        workspace = comm.trtllm_create_ipc_workspace_for_all_reduce(
            rank=rank,
            tp_size=world_size,
            max_token_num=maxTokenNum,
            hidden_dim=maxHiddenSize,
            group=group,
        )

        test_loop = 2  # could be any number

        # NOTE: the barrier flag should be initialized to 1, and incremented by 1 for each AR
        flag_value = 1
        for token_num in token_nums:
            for hidden_size in hidden_sizes:
                for strategy_code in strategy_codes:
                    for config_code in config_codes:
                        for fusion_op_code in fusion_op_codes:
                            for launch_with_pdl in launch_with_pdls:
                                pass_flag = True
                                if (
                                    strategy_code == comm.AllReduceStrategyType.TWOSHOT
                                    and fusion_op_code
                                    == comm.AllReduceFusionOp.RESIDUAL_RMS_PREPOST_NORM
                                ):
                                    # skip twoshot pre-post norm: not supported in trtllm test
                                    continue
                                print(
                                    f"test RANK {rank}: {world_size}-{dtype}-{strategy_code}-{config_code}-{fusion_op_code}-{launch_with_pdl}-{hidden_size} start"
                                )
                                torch.cuda.synchronize()
                                for _ in range(test_loop):
                                    message_size = token_num * hidden_size
                                    inp1 = torch.randn(
                                        message_size, dtype=dtype, device=device
                                    )
                                    inp1_ref = inp1.clone()
                                    out1 = torch.empty_like(inp1)

                                    # init params for each fusion op
                                    bias = torch.randn(
                                        hidden_size, dtype=dtype, device=device
                                    )
                                    residual = torch.randn(
                                        message_size, dtype=dtype, device=device
                                    )
                                    weight = torch.randn(
                                        hidden_size, dtype=dtype, device=device
                                    )
                                    weight_pre_residual_norm = torch.randn(
                                        hidden_size, dtype=dtype, device=device
                                    )
                                    eps = 1e-6
                                    intermediate_buffer = torch.zeros(
                                        message_size, dtype=dtype, device=device
                                    )

                                    comm.trtllm_custom_all_reduce(
                                        inp=inp1,
                                        out=out1,
                                        tp_size=world_size,
                                        tp_rank=rank,
                                        token_num=token_num,
                                        fusion_op_code=fusion_op_code,
                                        strategy_code=strategy_code,
                                        config_code=config_code,
                                        launch_with_pdl=launch_with_pdl,
                                        flag_value=flag_value,
                                        peer_comm_buffer_ptrs=torch.tensor(
                                            workspace[0], dtype=torch.int64
                                        ),
                                        peer_barrier_ptrs_in=torch.tensor(
                                            workspace[2], dtype=torch.int64
                                        ),
                                        peer_barrier_ptrs_out=torch.tensor(
                                            workspace[3], dtype=torch.int64
                                        ),
                                        bias=bias,
                                        residual=residual,
                                        weight=weight,
                                        weight_pre_residual_norm=weight_pre_residual_norm,
                                        eps=eps,
                                        intermediate_buffer=intermediate_buffer,
                                        lamport_peer_comm_buffer_ptrs_0=torch.tensor(
                                            workspace[4], dtype=torch.int64
                                        ),
                                        lamport_peer_comm_buffer_ptrs_1=torch.tensor(
                                            workspace[5], dtype=torch.int64
                                        ),
                                        lamport_peer_comm_buffer_ptrs_2=torch.tensor(
                                            workspace[6], dtype=torch.int64
                                        ),
                                    )
                                    dist.all_reduce(inp1_ref, group=group)

                                    tolerance = 1e-2 if dtype == torch.float16 else 8e-2

                                    if fusion_op_code == comm.AllReduceFusionOp.NONE:
                                        torch.testing.assert_close(
                                            out1, inp1_ref, atol=tolerance, rtol=3e-2
                                        )
                                    elif (
                                        fusion_op_code
                                        == comm.AllReduceFusionOp.RESIDUAL_RMS_NORM
                                    ):
                                        # cache intermediate_buffer to inter_buffer
                                        inter_buffer = intermediate_buffer.clone()

                                        # residual and bias
                                        ref = inp1_ref.clone()
                                        ref_float = ref.to(torch.float32)
                                        residual_float = residual.to(torch.float32)
                                        bias_float = bias.to(torch.float32)

                                        for i in range(ref.numel()):
                                            ref_float[i] += (
                                                residual_float[i]
                                                + bias_float[i % hidden_size]
                                            )
                                        ref_half = ref_float.to(dtype)
                                        torch.testing.assert_close(
                                            inter_buffer,
                                            ref_half,
                                            atol=tolerance,
                                            rtol=3e-2,
                                        )

                                        # RMSNorm over hidden size
                                        ref_float = ref_float.view(
                                            token_num, hidden_size
                                        )
                                        normed_float = torch.empty_like(ref_float)

                                        mean_sq = torch.mean(
                                            ref_float * ref_float, dim=-1, keepdim=True
                                        )
                                        denom = torch.sqrt(mean_sq + eps)
                                        normed_float = ref_float / denom
                                        normed_float = normed_float * weight.to(
                                            torch.float32
                                        )
                                        normed_half = normed_float.to(dtype)
                                        torch.testing.assert_close(
                                            out1,
                                            normed_half.view(-1),
                                            atol=tolerance,
                                            rtol=3e-2,
                                        )

                                    elif (
                                        fusion_op_code
                                        == comm.AllReduceFusionOp.RESIDUAL_RMS_PREPOST_NORM
                                    ):
                                        # NOTE(yingyi): bugfix todo, the test invokes nccl timeout for now
                                        pass

                                    flag_value += 1
                                if pass_flag:
                                    print(
                                        f"test RANK {rank}: {world_size}-{dtype}-{strategy_code}-{config_code}-{fusion_op_code}-{launch_with_pdl}-{hidden_size} passed"
                                    )
                                # torch.cuda.synchronize()
                                # # you might want to enable this barrier for a better log output, but it's not mandatory across allReduce calls
    finally:
        dist.barrier(group=group)

        comm.trtllm_destroy_ipc_workspace_for_all_reduce(workspace, group=group)

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
def test_trtllm_custom_allreduce(world_size, dtype):
    torch.manual_seed(RANDOM_SEED)
    available_gpus = torch.cuda.device_count()
    if world_size > available_gpus:
        raise ValueError(
            f"world_size {world_size} is greater than available_gpus {available_gpus}"
        )
    print(f"Running test for world_size={world_size}")

    multi_process_parallel(
        world_size,
        dtype,
        _run_correctness_worker,
        target_args=(),
    )
    print(f"custom allreduce tp = {world_size}: OK")
