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
maxSeqLen = 65536
hiddenSize = 64


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
            # 0,
            comm.AllReduceStrategyConfig.USE_MEMCPY,
            comm.AllReduceStrategyConfig.PUSH_MODE,
        ]
        fusion_op_codes = [
            0,
            comm.AllReduceFusionOp.NONE,
            comm.AllReduceFusionOp.RESIDUAL_RMS_NORM,
            comm.AllReduceFusionOp.RESIDUAL_RMS_PREPOST_NORM,
            # below are not enabled in trtllm test, skip for now
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
        ipc_handles = comm.trtllm_create_ipc_buffer_for_all_reduce(
            rank, world_size, maxSeqLen, hiddenSize, group=group
        )

        test_loop = 1

        flag_value = 0
        for token_num in token_nums:
            for strategy_code in strategy_codes:
                for config_code in config_codes:
                    for fusion_op_code in fusion_op_codes:
                        for launch_with_pdl in launch_with_pdls:
                            print(
                                f"test RANK {rank}: {world_size}-{dtype}-{strategy_code}-{config_code}-{fusion_op_code}-{launch_with_pdl}-{flag_value} start"
                            )
                            for _ in range(test_loop):
                                message_size = token_num * hiddenSize
                                inp1 = torch.rand(
                                    message_size, dtype=dtype, device=device
                                )
                                inp1_ref = inp1.clone()
                                out1 = torch.empty_like(inp1)

                                # lamport init to negative zero
                                comm.trtllm_lamport_initialize_all(
                                    ipc_handles[4][rank],
                                    ipc_handles[5][rank],
                                    ipc_handles[6][rank],
                                    message_size,
                                    dtype,
                                )
                                flag_value += 1

                                # init params for each fusion op
                                bias = torch.rand(
                                    hiddenSize, dtype=dtype, device=device
                                )
                                residual = torch.rand(
                                    message_size, dtype=dtype, device=device
                                )
                                weight = torch.rand(
                                    hiddenSize, dtype=dtype, device=device
                                )
                                weight_pre_residual_norm = None
                                eps = None
                                intermediate_buffer = torch.rand(
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
                                        ipc_handles[0], dtype=torch.int64
                                    ),
                                    peer_barrier_ptrs_in=torch.tensor(
                                        ipc_handles[2], dtype=torch.int64
                                    ),
                                    peer_barrier_ptrs_out=torch.tensor(
                                        ipc_handles[3], dtype=torch.int64
                                    ),
                                    bias=bias,
                                    residual=residual,
                                    weight=weight,
                                    weight_pre_residual_norm=weight_pre_residual_norm,
                                    eps=eps,
                                    intermediate_buffer=intermediate_buffer,
                                    lamport_peer_comm_buffer_ptrs_0=torch.tensor(
                                        ipc_handles[4], dtype=torch.int64
                                    ),
                                    lamport_peer_comm_buffer_ptrs_1=torch.tensor(
                                        ipc_handles[5], dtype=torch.int64
                                    ),
                                    lamport_peer_comm_buffer_ptrs_2=torch.tensor(
                                        ipc_handles[6], dtype=torch.int64
                                    ),
                                )
                                dist.all_reduce(inp1_ref, group=group)

                                if fusion_op_code == comm.AllReduceFusionOp.NONE:
                                    torch.testing.assert_close(
                                        out1, inp1_ref, atol=1e-4, rtol=3e-2
                                    )
                                elif (
                                    fusion_op_code
                                    == comm.AllReduceFusionOp.RESIDUAL_RMS_NORM
                                ):
                                    # # Step 1: Copy intermediate_buffer to ref
                                    # inter_buffer = intermediate_buffer.clone()

                                    # # Step 2: Add residual and bias
                                    # has_bias = bias is not None
                                    # has_affine = weight is not None
                                    # eps_value = 1e-5 if eps is None else eps

                                    # # Convert to float32 for precision
                                    # ref = ref.to(torch.float32)
                                    # residual_f = residual.to(torch.float32)
                                    # if has_bias:
                                    #     bias_f = bias.to(torch.float32)

                                    # for i in range(ref.numel()):
                                    #     ref[i] += residual_f[i]
                                    #     if has_bias:
                                    #         ref[i] += bias_f[i % hiddenSize]

                                    # # Step 3: RMSNorm over hidden size
                                    # ref = ref.view(token_num, hiddenSize)
                                    # normed = torch.empty_like(ref)

                                    # for i in range(token_num):
                                    #     vec = ref[i]
                                    #     mean_sq = torch.mean(vec * vec)
                                    #     denom = torch.sqrt(mean_sq + eps_value)
                                    #     if has_affine:
                                    #         normed[i] = (vec / denom) * weight.to(torch.float32)
                                    #     else:
                                    #         normed[i] = vec / denom

                                    # # Step 4: Validate output
                                    # torch.testing.assert_close(out1.to(torch.float32), normed.view(-1), atol=1e-2, rtol=1e-2)
                                    pass
                                elif fusion_op_code == comm.AllReduceFusionOp.RESIDUAL_RMS_PREPOST_NORM:
                                    pass
                            print(
                                f"test RANK {rank}: {world_size}-{dtype}-{strategy_code}-{config_code}-{fusion_op_code}-{launch_with_pdl}-{flag_value} passed"
                            )
    finally:
        dist.barrier(group=group)

        # todo(yingyi): should we have a comm manager for all tllm allreduce
        # if custom_ptr is not None:
        #     comm.dispose(custom_ptr)

        # todo(yingyi): free ipc handles
        for ipc_handle in ipc_handles:
            comm.free_shared_buffer(ipc_handle, group)

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

    # add py interface binding tests
    # comm.trtllm_lamport_initialize_all(
    #     torch.empty(1024, dtype=torch.float32, device="cuda").data_ptr(),
    #     torch.empty(1024, dtype=torch.float32, device="cuda").data_ptr(),
    #     torch.empty(1024, dtype=torch.float32, device="cuda").data_ptr(),
    #     1024,
    #     torch.float32,
    # )
    # comm.trtllm_custom_all_reduce(
    #     torch.empty(1024, dtype=torch.float32, device="cuda"),
    #     torch.empty(1024, dtype=torch.float32, device="cuda"),
    #     2,
    #     0,
    #     1024,
    #     0,
    #     0,
    #     0,
    #     True,
    #     0,
    #     torch.empty(1024, dtype=torch.float32, device="cuda"),
    #     torch.empty(1024, dtype=torch.float32, device="cuda"),
    #     torch.empty(1024, dtype=torch.float32, device="cuda"),
    #     None,
    #     None,
    #     None,
    #     None,
    #     None,
    #     None,
    #     None,
    #     None,
    # )

    # todo: add real tests
    set_log_level("info")
    test_trtllm_custom_allreduce(2, torch.float16)
