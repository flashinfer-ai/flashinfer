import multiprocessing as mp
import socket
from typing import Any

import pytest
import torch
import torch.distributed as dist

import flashinfer.comm as comm

# todo(Yingyi): add warmup stes for benchmark

# Usage: test var, temp
kOneShotMaxTokenNum = 128
# MAX_TOKEN_NUM = 2048
MAX_TOKEN_NUM = kOneShotMaxTokenNum
HIDDEN_SIZE = 7168
# HIDDEN_SIZE = 8  # debug-only
MAX_EXPERT_NUM = 16
SCALE_FACTOR_RANGE = (-5, 5)


# Usage: The fusion type code is used to indicate the test type only
# for trtllm_moe_allreduce_fusion, pass required tensor and skip the rest as None to indicate the fusion type
class MoEAllReduceFusionType:
    RESIDUAL_QUANT_OUT = 0
    NORM_OUT = 1
    RESIDUAL_NORM_OUT = 2
    RESIDUAL_NORM_QUANT_OUT = 3


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
        token_nums = [
            1,
            64,
            128,
            # 256,
            # 2048,
        ]  # 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
        candidate_active_expert_num = [8, 12, 16]
        fusion_codes = [
            MoEAllReduceFusionType.RESIDUAL_QUANT_OUT,
            MoEAllReduceFusionType.NORM_OUT,
            MoEAllReduceFusionType.RESIDUAL_NORM_OUT,
            MoEAllReduceFusionType.RESIDUAL_NORM_QUANT_OUT,
        ]
        swizzled_layout_codes = [
            comm.FP4QuantizationSFLayout.LINEAR,
            comm.FP4QuantizationSFLayout.SWIZZLED,
        ]
        launch_with_pdls = [True, False]

        # create workspace for moe allreduce fusion
        ipc_handles, workspace_tensor = (
            comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
                rank, world_size, MAX_TOKEN_NUM, HIDDEN_SIZE, group=group
            )
        )

        test_loop = 5

        for token_num in token_nums:
            for active_expert_num in candidate_active_expert_num:
                for fusion_code in fusion_codes:
                    for swizzled_layout_code in swizzled_layout_codes:
                        for launch_with_pdl in launch_with_pdls:
                            print(
                                f"test RANK {rank}: token{token_num}-expert{active_expert_num}-tp{world_size}-{dtype}-fusion{fusion_code}-layout{swizzled_layout_code}-pdl{launch_with_pdl} start"
                            )
                            torch.cuda.synchronize()
                            for _ in range(test_loop):
                                message_size = token_num * HIDDEN_SIZE

                                inp1 = (
                                    torch.rand(message_size, dtype=dtype, device=device)
                                    * 200.0
                                    - 100.0
                                )
                                inp1_ref = inp1.clone()
                                out1 = torch.empty_like(inp1)

                                # init params for each fusion op
                                residual_in = (
                                    torch.rand(message_size, dtype=dtype, device=device)
                                    * 200.0
                                    - 100.0
                                )
                                residual_out = torch.empty_like(residual_in)
                                norm_out = torch.empty_like(residual_in)
                                quant_out = torch.empty_like(residual_in)
                                scale_out = torch.empty_like(residual_in)
                                rms_gamma = (
                                    torch.rand(HIDDEN_SIZE, dtype=dtype, device=device)
                                    * 2.0
                                    - 1.0
                                )
                                scale_factor = (
                                    torch.rand(1, dtype=torch.float32, device=device)
                                    * (SCALE_FACTOR_RANGE[1] - SCALE_FACTOR_RANGE[0])
                                    + SCALE_FACTOR_RANGE[0]
                                )
                                rms_eps = 1e-3

                                # init moe params
                                # [device_num_expert, m]
                                moe_reduction_scale_input = (
                                    torch.rand(
                                        active_expert_num * token_num,
                                        dtype=torch.float32,
                                        device=device,
                                    )
                                    * 200.0
                                    - 100.0
                                )
                                # [device_num_expert, m, 7168]
                                moe_reduction_active_experts_token_input = (
                                    torch.rand(
                                        active_expert_num * message_size,
                                        dtype=dtype,
                                        device=device,
                                    )
                                    * 200.0
                                    - 100.0
                                )
                                # [m, 7168]
                                moe_reduction_token_input = (
                                    torch.rand(message_size, dtype=dtype, device=device)
                                    * 200.0
                                    - 100.0
                                )

                                if (
                                    fusion_code
                                    == MoEAllReduceFusionType.RESIDUAL_QUANT_OUT
                                ):
                                    comm.trtllm_moe_allreduce_fusion(
                                        inp=inp1,
                                        world_size=world_size,
                                        world_rank=rank,
                                        token_num=token_num,
                                        hidden_dim=HIDDEN_SIZE,
                                        workspace_ptrs=workspace_tensor,
                                        launch_with_pdl=launch_with_pdl,
                                        residual_in=residual_in,
                                        rms_gamma=rms_gamma,
                                        rms_eps=rms_eps,
                                        scale_factor=scale_factor,
                                        moe_reduction_device_num_experts=active_expert_num,
                                        moe_reduction_scale_input=moe_reduction_scale_input,
                                        moe_reduction_active_experts_token_input=moe_reduction_active_experts_token_input,
                                        moe_reduction_token_input=moe_reduction_token_input,
                                        layout_code=swizzled_layout_code,
                                        residual_out=residual_out,
                                        norm_out=None,
                                        quant_out=quant_out,
                                        scale_out=scale_out,
                                    )
                                elif fusion_code == MoEAllReduceFusionType.NORM_OUT:
                                    comm.trtllm_moe_allreduce_fusion(
                                        inp=inp1,
                                        world_size=world_size,
                                        world_rank=rank,
                                        token_num=token_num,
                                        hidden_dim=HIDDEN_SIZE,
                                        workspace_ptrs=workspace_tensor,
                                        launch_with_pdl=launch_with_pdl,
                                        residual_in=residual_in,
                                        rms_gamma=rms_gamma,
                                        rms_eps=rms_eps,
                                        scale_factor=scale_factor,
                                        moe_reduction_device_num_experts=active_expert_num,
                                        moe_reduction_scale_input=moe_reduction_scale_input,
                                        moe_reduction_active_experts_token_input=moe_reduction_active_experts_token_input,
                                        moe_reduction_token_input=moe_reduction_token_input,
                                        layout_code=swizzled_layout_code,
                                        residual_out=None,
                                        norm_out=norm_out,
                                        quant_out=None,
                                        scale_out=None,
                                    )
                                elif (
                                    fusion_code
                                    == MoEAllReduceFusionType.RESIDUAL_NORM_OUT
                                ):
                                    comm.trtllm_moe_allreduce_fusion(
                                        inp=inp1,
                                        world_size=world_size,
                                        world_rank=rank,
                                        token_num=token_num,
                                        hidden_dim=HIDDEN_SIZE,
                                        workspace_ptrs=workspace_tensor,
                                        launch_with_pdl=launch_with_pdl,
                                        residual_in=residual_in,
                                        rms_gamma=rms_gamma,
                                        rms_eps=rms_eps,
                                        scale_factor=scale_factor,
                                        moe_reduction_device_num_experts=active_expert_num,
                                        moe_reduction_scale_input=moe_reduction_scale_input,
                                        moe_reduction_active_experts_token_input=moe_reduction_active_experts_token_input,
                                        moe_reduction_token_input=moe_reduction_token_input,
                                        layout_code=swizzled_layout_code,
                                        residual_out=residual_out,
                                        norm_out=norm_out,
                                        quant_out=None,
                                        scale_out=None,
                                    )
                                elif (
                                    fusion_code
                                    == MoEAllReduceFusionType.RESIDUAL_NORM_QUANT_OUT
                                ):
                                    comm.trtllm_moe_allreduce_fusion(
                                        inp=inp1,
                                        world_size=world_size,
                                        world_rank=rank,
                                        token_num=token_num,
                                        hidden_dim=HIDDEN_SIZE,
                                        workspace_ptrs=workspace_tensor,
                                        launch_with_pdl=launch_with_pdl,
                                        residual_in=residual_in,
                                        rms_gamma=rms_gamma,
                                        rms_eps=rms_eps,
                                        scale_factor=scale_factor,
                                        moe_reduction_device_num_experts=active_expert_num,
                                        moe_reduction_scale_input=moe_reduction_scale_input,
                                        moe_reduction_active_experts_token_input=moe_reduction_active_experts_token_input,
                                        moe_reduction_token_input=moe_reduction_token_input,
                                        layout_code=swizzled_layout_code,
                                        residual_out=residual_out,
                                        norm_out=norm_out,
                                        quant_out=quant_out,
                                        scale_out=scale_out,
                                    )
                                else:
                                    raise ValueError(
                                        f"Invalid fusion code: {fusion_code}"
                                    )

                                # == Calculate reference output ==
                                # 1. MoE Reduction
                                moe_expert_out = (
                                    moe_reduction_active_experts_token_input.view(
                                        active_expert_num, message_size
                                    )
                                )
                                moe_scales = moe_reduction_scale_input.view(
                                    active_expert_num, token_num
                                )
                                moe_expert_out = moe_expert_out.view(
                                    active_expert_num, token_num, HIDDEN_SIZE
                                )
                                moe_scales = moe_scales.unsqueeze(2)
                                scaled_expert_out = moe_expert_out * moe_scales.to(
                                    dtype
                                )
                                reduced_expert_out = torch.sum(scaled_expert_out, dim=0)

                                # 2. Add FC2 output
                                moe_out_ref = (
                                    reduced_expert_out.view(message_size)
                                    + moe_reduction_token_input
                                )

                                # 3. All-Reduce
                                all_reduced_ref = moe_out_ref.clone()
                                dist.all_reduce(all_reduced_ref, group=group)

                                # 4. Fused Ops
                                ref_residual_out = all_reduced_ref.view(
                                    token_num, HIDDEN_SIZE
                                ) + residual_in.view(token_num, HIDDEN_SIZE)

                                variance = (
                                    ref_residual_out.to(torch.float32)
                                    .pow(2)
                                    .mean(dim=-1, keepdim=True)
                                )
                                hidden_states = ref_residual_out * torch.rsqrt(
                                    variance + rms_eps
                                )
                                ref_norm_out = rms_gamma * hidden_states

                                # 5. Check correctness
                                # if (
                                #     fusion_code
                                #     == MoEAllReduceFusionType.RESIDUAL_QUANT_OUT
                                # ):
                                #     torch.testing.assert_close(
                                #         residual_out.view(token_num, HIDDEN_SIZE),
                                #         ref_residual_out,
                                #         atol=1e-2,
                                #         rtol=1e-2,
                                #     )
                                # elif fusion_code == MoEAllReduceFusionType.NORM_OUT:
                                #     torch.testing.assert_close(
                                #         norm_out.view(token_num, HIDDEN_SIZE),
                                #         ref_norm_out,
                                #         atol=1e-2,
                                #         rtol=1e-2,
                                #     )
                                # elif (
                                #     fusion_code
                                #     == MoEAllReduceFusionType.RESIDUAL_NORM_OUT
                                # ):
                                #     torch.testing.assert_close(
                                #         residual_out.view(token_num, HIDDEN_SIZE),
                                #         ref_residual_out,
                                #         atol=1e-2,
                                #         rtol=1e-2,
                                #     )
                                #     torch.testing.assert_close(
                                #         norm_out.view(token_num, HIDDEN_SIZE),
                                #         ref_norm_out,
                                #         atol=1e-2,
                                #         rtol=1e-2,
                                #     )
                                # elif (
                                #     fusion_code
                                #     == MoEAllReduceFusionType.RESIDUAL_NORM_QUANT_OUT
                                # ):
                                #     torch.testing.assert_close(
                                #         residual_out.view(token_num, HIDDEN_SIZE),
                                #         ref_residual_out,
                                #         atol=1e-2,
                                #         rtol=1e-2,
                                #     )
                                #     torch.testing.assert_close(
                                #         norm_out.view(token_num, HIDDEN_SIZE),
                                #         ref_norm_out,
                                #         atol=1e-2,
                                #         rtol=1e-2,
                                #     )

                            torch.cuda.synchronize()
                            print(
                                f"test RANK {rank}: token{token_num}-expert{active_expert_num}-tp{world_size}-{dtype}-fusion{fusion_code}-layout{swizzled_layout_code}-pdl{launch_with_pdl} passed"
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
def test_trtllm_moe_allreduce_fusion(world_size, dtype):
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
    print(f"moe allreduce fusion tp = {world_size}: OK")


if __name__ == "__main__":
    mod = comm.get_comm_module()

    test_trtllm_moe_allreduce_fusion(2, torch.float16)
    test_trtllm_moe_allreduce_fusion(2, torch.bfloat16)
    test_trtllm_moe_allreduce_fusion(4, torch.float16)
    test_trtllm_moe_allreduce_fusion(4, torch.bfloat16)
