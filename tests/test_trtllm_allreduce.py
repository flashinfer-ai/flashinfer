import multiprocessing as mp
import socket
from typing import Any

import pytest
import torch
import torch.distributed as dist

import flashinfer.comm as comm
from flashinfer.utils import set_log_level

# todo: temp for test
maxBatchSize = 1
maxBeamWidth = 3
maxSeqLen = 65536 # max sequence length for all reduce
hiddenSize = 64 # max hidden size for all reduce


def _run_correctness_worker(world_size, rank, dtype, distributed_init_port):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    distributed_init_method = f"tcp://localhost:{distributed_init_port}"
    dist.init_process_group(
        backend="nccl",  # todo: do we need to support other backends?
        init_method=distributed_init_method,
        rank=rank,
        world_size=world_size,
    )
    group = dist.group.WORLD

    try:
        device = torch.device(f"cuda:{rank}")
        token_nums = [
            4096,
            8192,
        ]
        strategy_codes = [
            comm.AllReduceStrategyType.ONESHOT,
            comm.AllReduceStrategyType.TWOSHOT,
        ]
        config_codes = [
            0,
            comm.AllReduceStrategyConfig.USE_MEMCPY,
            comm.AllReduceStrategyConfig.PUSH_MODE,
        ]
        fusion_op_codes = [
            comm.AllReduceFusionOp.NONE,
            comm.AllReduceFusionOp.RESIDUAL_RMS_NORM,

            # NOTE(yingyi): bug - nccl timeout on pre-post norm
            # comm.AllReduceFusionOp.RESIDUAL_RMS_PREPOST_NORM,

            # below are not enabled for custom all-reduce in trtllm test, skip
            # comm.AllReduceFusionOp.LAST_PROCESS_FOR_UB,
            # comm.AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8,
            # comm.AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4,
            # comm.AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_FP8,
            # comm.AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4,
            # comm.AllReduceFusionOp.MOE_ALLREDUCE_RESIDUAL_RMS_NORM,
        ]
        launch_with_pdls = [True, False]

        # create ipc memory
        # todo(yingyi): lamport should be init only when can_access_peer is true, is it true default?
        # init per world_size?
        workspace = comm.trtllm_create_ipc_workspace_for_all_reduce(
            rank, world_size, maxSeqLen, hiddenSize, group=group
        )

        test_loop = 1

        # NOTE: the barrier flag should be initialized to 1, and incremented by 1 for each AR
        flag_value = 1
        for token_num in token_nums:
            for strategy_code in strategy_codes:
                for config_code in config_codes:
                    for fusion_op_code in fusion_op_codes:
                        for launch_with_pdl in launch_with_pdls:
                            if (
                                strategy_code == comm.AllReduceStrategyType.TWOSHOT
                                and fusion_op_code
                                == comm.AllReduceFusionOp.RESIDUAL_RMS_PREPOST_NORM
                            ):
                                # skip twoshot pre-post norm: not supported in trtllm test
                                continue
                            print(
                                f"test RANK {rank}: {world_size}-{dtype}-{strategy_code}-{config_code}-{fusion_op_code}-{launch_with_pdl} start"
                            )
                            for _ in range(test_loop):
                                message_size = token_num * hiddenSize
                                inp1 = torch.randn(
                                    message_size, dtype=dtype, device=device
                                )
                                inp1_ref = inp1.clone()
                                out1 = torch.empty_like(inp1)

                                # lamport init to negative zero
                                comm.trtllm_lamport_initialize_all(
                                    workspace[4][rank],
                                    workspace[5][rank],
                                    workspace[6][rank],
                                    message_size,
                                    dtype,
                                )

                                # init params for each fusion op
                                bias = (
                                    torch.randn(hiddenSize, dtype=dtype, device=device)
                                )
                                residual = (
                                    torch.randn(message_size, dtype=dtype, device=device)
                                )
                                weight = (
                                    torch.randn(hiddenSize, dtype=dtype, device=device)
                                )
                                weight_pre_residual_norm = (
                                    torch.randn(hiddenSize, dtype=dtype, device=device)
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
                                # print(
                                #     f"test RANK {rank}: {world_size}-{dtype}-{strategy_code}-{config_code}-{fusion_op_code}-{launch_with_pdl}-{flag_value} test AR done"
                                # )
                                dist.all_reduce(inp1_ref, group=group)
                                # print(
                                #     f"test RANK {rank}: {world_size}-{dtype}-{strategy_code}-{config_code}-{fusion_op_code}-{launch_with_pdl}-{flag_value} ref AR done"
                                # )

                                if fusion_op_code == comm.AllReduceFusionOp.NONE:
                                    torch.testing.assert_close(
                                        out1, inp1_ref, atol=1e-2, rtol=3e-2
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
                                            + bias_float[i % hiddenSize]
                                        )
                                    ref_half = ref_float.to(dtype)

                                    torch.testing.assert_close(
                                        inter_buffer, ref_half, atol=1e-2, rtol=3e-2
                                    )

                                    # RMSNorm over hidden size
                                    ref_float = ref_float.view(token_num, hiddenSize)
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
                                        out1, normed_half.view(-1), atol=1e-2, rtol=1e-2
                                    )

                                elif (
                                    fusion_op_code
                                    == comm.AllReduceFusionOp.RESIDUAL_RMS_PREPOST_NORM
                                ):
                                    # NOTE(yingyi): bugfix todo, the test invokes nccl timeout for now
                                    pass

                                    # ref_float = inp1_ref.clone()
                                    # ref_float = ref_float.to(torch.float32)
                                    # residual_float = residual.to(torch.float32)
                                    # bias_float = bias.to(torch.float32)

                                    # for i in range(ref.numel()):
                                    #     ref_float[i] += (
                                    #         residual_float[i]
                                    #         + bias_float[i % hiddenSize]
                                    #     )
                                    # ref_half = ref_float.to(dtype)

                                flag_value += 1
                            print(
                                f"test RANK {rank}: {world_size}-{dtype}-{strategy_code}-{config_code}-{fusion_op_code}-{launch_with_pdl} passed"
                            )
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
        assert (
            procs[i].exitcode == 0
        ), f"Process {i} failed with exit code {procs[i].exitcode}"


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_trtllm_custom_allreduce(world_size, dtype):
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


if __name__ == "__main__":
    # compile tests
    mod = comm.get_comm_module()

    # pass real tests
    # test_trtllm_custom_allreduce(2, torch.float16)
    test_trtllm_custom_allreduce(2, torch.bfloat16)
    # test_trtllm_custom_allreduce(4, torch.float16)
    test_trtllm_custom_allreduce(4, torch.bfloat16)
