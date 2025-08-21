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
MAX_TOKEN_NUM = 2048
HIDDEN_SIZE = 7168
MAX_EXPERT_NUM = 16
SF_VEC_SIZE = 16

# temp var
SCALE_FACTOR_RANGE = (-1, 1)


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
            256,
            2048,
        ]  # 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
        candidate_active_expert_num = [8, 12, 16]
        # candidate_active_expert_num = [1]  # debug-only
        swizzled_layout_codes = [
            comm.QuantizationSFLayout.LINEAR,
            comm.QuantizationSFLayout.SWIZZLED_128x4,
            comm.QuantizationSFLayout.SWIZZLED_8x4,
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
                for swizzled_layout_code in swizzled_layout_codes:
                    for launch_with_pdl in launch_with_pdls:
                        dist.barrier(group=group)
                        test_passed = True
                        print(
                            f"test RANK {rank}: token{token_num}-expert{active_expert_num}-tp{world_size}-{dtype}-layout{swizzled_layout_code}-pdl{launch_with_pdl} start"
                        )
                        dist.barrier(group=group)
                        torch.cuda.synchronize()
                        for _ in range(test_loop):
                            message_size = token_num * HIDDEN_SIZE

                            residual_in = torch.randn(
                                message_size, dtype=dtype, device=device
                            )
                            residual_in_clone = residual_in.clone()

                            moe_allreduce_out = torch.zeros(
                                message_size, dtype=dtype, device=device
                            )
                            residual_out = torch.empty_like(residual_in)
                            norm_out = torch.empty_like(residual_in)
                            quant_out = torch.empty(
                                message_size // 4, dtype=dtype, device=device
                            )  # quant: fp16/bf16 -> fp4, reference: cpp/tensorrt_llm/thop/allreduceOp.cpp:L487

                            scale_out = None
                            assert HIDDEN_SIZE % SF_VEC_SIZE == 0, (
                                "HIDDEN_SIZE must be divisible by SF_VEC_SIZE"
                            )
                            if (
                                swizzled_layout_code
                                == comm.QuantizationSFLayout.SWIZZLED_128x4
                            ):
                                padded_message_size = (
                                    comm.compute_fp4_swizzled_layout_sf_size(
                                        token_num, HIDDEN_SIZE // SF_VEC_SIZE
                                    )
                                )
                                scale_out = torch.empty(
                                    padded_message_size, dtype=dtype, device=device
                                )
                            else:
                                scale_out = torch.empty(
                                    message_size // SF_VEC_SIZE,
                                    dtype=dtype,
                                    device=device,
                                )

                            rms_gamma = torch.randn(
                                HIDDEN_SIZE, dtype=dtype, device=device
                            )
                            scale_factor = (
                                torch.rand(1, dtype=torch.float32, device=device)
                                * (SCALE_FACTOR_RANGE[1] - SCALE_FACTOR_RANGE[0])
                                + SCALE_FACTOR_RANGE[0]
                            )
                            rms_eps = 1e-3
                            scale_factor_float = scale_factor.item()

                            # init moe params
                            # [device_num_expert, m]
                            moe_reduction_scale_input = torch.randn(
                                active_expert_num * token_num,
                                dtype=torch.float32,
                                device=device,
                            )

                            moe_reduction_scale_input_clone = (
                                moe_reduction_scale_input.clone()
                            )

                            # [device_num_expert, m, 7168]
                            moe_reduction_active_experts_token_input = torch.randn(
                                active_expert_num * message_size,
                                dtype=dtype,
                                device=device,
                            )
                            moe_reduction_active_experts_token_input_clone = (
                                moe_reduction_active_experts_token_input.clone()
                            )
                            # [m, 7168]
                            moe_reduction_token_input = torch.randn(
                                message_size, dtype=dtype, device=device
                            )
                            moe_reduction_token_input_clone = (
                                moe_reduction_token_input.clone()
                            )

                            # == Calculate reference output ==
                            # 1. MoE Reduction
                            moe_expert_out = (
                                moe_reduction_active_experts_token_input_clone.view(
                                    active_expert_num, token_num, HIDDEN_SIZE
                                ).to(torch.float32)
                            )
                            moe_scales = moe_reduction_scale_input_clone.view(
                                active_expert_num, token_num
                            ).to(torch.float32)
                            moe_scales = moe_scales.unsqueeze(
                                2
                            )  # [active_expert_num, token_num, 1]
                            scaled_expert_out = moe_expert_out * moe_scales.to(
                                torch.float32
                            )  # [active_expert_num, token_num, HIDDEN_SIZE]
                            reduced_expert_out = torch.sum(
                                scaled_expert_out, dim=0
                            )  # [token_num, HIDDEN_SIZE]

                            # 2. Add FC2 output
                            moe_out_ref = (
                                reduced_expert_out
                                + moe_reduction_token_input_clone.view(
                                    token_num, HIDDEN_SIZE
                                ).to(torch.float32)
                            )  # [token_num, HIDDEN_SIZE]

                            # 3. All-Reduce
                            moe_allreduce_ref = moe_out_ref.clone().to(dtype)
                            dist.all_reduce(moe_allreduce_ref, group=group)
                            moe_allreduce_ref = moe_allreduce_ref.to(torch.float32)

                            # 4. Fused Ops
                            ref_residual_out = (
                                moe_allreduce_ref
                                + residual_in_clone.view(token_num, HIDDEN_SIZE).to(
                                    torch.float32
                                )
                            )

                            variance = (
                                ref_residual_out.to(torch.float32)
                                .pow(2)
                                .mean(dim=-1, keepdim=True)
                            )
                            hidden_states = ref_residual_out * torch.rsqrt(
                                variance + rms_eps
                            )
                            ref_norm_out = rms_gamma.to(torch.float32) * hidden_states

                            # 5. Run kernel
                            # warmup
                            s = torch.cuda.Stream()
                            s.wait_stream(torch.cuda.current_stream())
                            with torch.cuda.stream(s):
                                for _ in range(3):  # Multiple warmup iterations
                                    comm.trtllm_moe_allreduce_fusion(
                                        world_size=world_size,
                                        world_rank=rank,
                                        token_num=token_num,
                                        hidden_dim=HIDDEN_SIZE,
                                        workspace_ptrs=workspace_tensor,
                                        launch_with_pdl=launch_with_pdl,
                                        residual_in=residual_in,
                                        rms_gamma=rms_gamma,
                                        rms_eps=rms_eps,
                                        scale_factor=scale_factor_float,
                                        moe_reduction_device_num_experts=active_expert_num,
                                        moe_reduction_scale_input=moe_reduction_scale_input,
                                        moe_reduction_active_experts_token_input=moe_reduction_active_experts_token_input,
                                        moe_reduction_token_input=moe_reduction_token_input,
                                        layout_code=swizzled_layout_code,
                                        moe_allreduce_out=moe_allreduce_out,
                                        residual_out=residual_out,
                                        norm_out=norm_out,
                                        quant_out=quant_out,
                                        scale_out=scale_out,
                                    )
                            torch.cuda.current_stream().wait_stream(s)
                            torch.cuda.synchronize()  # Ensure warmup is complete

                            # capture
                            g = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(g):
                                for _ in range(3):  # Multiple iterations in graph
                                    comm.trtllm_moe_allreduce_fusion(
                                        world_size=world_size,
                                        world_rank=rank,
                                        token_num=token_num,
                                        hidden_dim=HIDDEN_SIZE,
                                        workspace_ptrs=workspace_tensor,
                                        launch_with_pdl=launch_with_pdl,
                                        residual_in=residual_in,
                                        rms_gamma=rms_gamma,
                                        rms_eps=rms_eps,
                                        scale_factor=scale_factor_float,
                                        moe_reduction_device_num_experts=active_expert_num,
                                        moe_reduction_scale_input=moe_reduction_scale_input,
                                        moe_reduction_active_experts_token_input=moe_reduction_active_experts_token_input,
                                        moe_reduction_token_input=moe_reduction_token_input,
                                        layout_code=swizzled_layout_code,
                                        moe_allreduce_out=moe_allreduce_out,
                                        residual_out=residual_out,
                                        norm_out=norm_out,
                                        quant_out=quant_out,
                                        scale_out=scale_out,
                                    )

                            # replay
                            g.replay()

                            # match shape
                            moe_allreduce_out = moe_allreduce_out.view(
                                token_num, HIDDEN_SIZE
                            )
                            residual_out = residual_out.view(token_num, HIDDEN_SIZE)
                            norm_out = norm_out.view(token_num, HIDDEN_SIZE)

                            torch.cuda.synchronize()

                            # 6. Check correctness
                            tolerance = 8e-2 if dtype == torch.float16 else 8e-1
                            # 6.1 Check allreduce_out
                            if not torch.allclose(
                                moe_allreduce_out.to(torch.float32),
                                moe_allreduce_ref,
                                atol=tolerance,
                                rtol=1e-2,
                            ):
                                test_passed = False
                                print(f"Rank {rank} moe_allreduce_out mismatch")
                                print(f"moe_allreduce_out: {moe_allreduce_out}")
                                print(f"moe_allreduce_ref: {moe_allreduce_ref}")
                                # Print max diff elements for allreduce_out
                                max_diff = torch.max(
                                    torch.abs(
                                        moe_allreduce_out.to(torch.float32)
                                        - moe_allreduce_ref
                                    )
                                )
                                max_diff_idx = torch.argmax(
                                    torch.abs(
                                        moe_allreduce_out.to(torch.float32)
                                        - moe_allreduce_ref
                                    )
                                )
                                print(
                                    f"Rank {rank} moe_allreduce_out max diff: {max_diff}"
                                )
                                print(
                                    f"Rank {rank} moe_allreduce_out max diff idx: {max_diff_idx}"
                                )
                                print(
                                    f"Rank {rank} moe_allreduce_out value at max diff: {moe_allreduce_out.view(-1)[max_diff_idx]}"
                                )
                                print(
                                    f"Rank {rank} moe_allreduce_out ref value at max diff: {moe_allreduce_ref.view(-1)[max_diff_idx]}"
                                )

                            torch.testing.assert_close(
                                moe_allreduce_out.to(torch.float32),
                                moe_allreduce_ref,
                                atol=tolerance,
                                rtol=1e-2,
                            )

                            # 6.2 Check residual_out
                            if not torch.allclose(
                                residual_out.to(torch.float32),
                                ref_residual_out,
                                atol=tolerance,
                                rtol=1e-2,
                            ):
                                test_passed = False
                                print(f"Rank {rank} residual_out mismatch")
                                print(f"residual_out: {residual_out}")
                                print(f"ref_residual_out: {ref_residual_out}")
                                # Print max diff elements for residual_out
                                max_diff = torch.max(
                                    torch.abs(
                                        residual_out.to(torch.float32)
                                        - ref_residual_out
                                    )
                                )
                                max_diff_idx = torch.argmax(
                                    torch.abs(
                                        residual_out.to(torch.float32)
                                        - ref_residual_out
                                    )
                                )
                                print(f"Rank {rank} residual_out max diff: {max_diff}")
                                print(
                                    f"Rank {rank} residual_out max diff idx: {max_diff_idx}"
                                )
                                print(
                                    f"Rank {rank} residual_out value at max diff: {residual_out.view(-1)[max_diff_idx]}"
                                )
                                print(
                                    f"Rank {rank} residual_out ref value at max diff: {ref_residual_out.view(-1)[max_diff_idx]}"
                                )
                            torch.testing.assert_close(
                                residual_out.to(torch.float32),
                                ref_residual_out,
                                atol=tolerance,
                                rtol=1e-2,
                            )
                            # 6.3 Check norm_out
                            if not torch.allclose(
                                norm_out.to(torch.float32),
                                ref_norm_out,
                                atol=tolerance,
                                rtol=1e-2,
                            ):
                                test_passed = False
                                print(f"Rank {rank} norm_out mismatch")
                                print(f"norm_out: {norm_out}")
                                print(f"ref_norm_out: {ref_norm_out}")
                                # Print max diff elements for norm_out
                                max_diff = torch.max(
                                    torch.abs(norm_out.to(torch.float32) - ref_norm_out)
                                )
                                max_diff_idx = torch.argmax(
                                    torch.abs(norm_out.to(torch.float32) - ref_norm_out)
                                )
                                print(f"Rank {rank} norm_out max diff: {max_diff}")
                                print(
                                    f"Rank {rank} norm_out max diff idx: {max_diff_idx}"
                                )
                                print(
                                    f"Rank {rank} norm_out value at max diff: {norm_out.view(-1)[max_diff_idx]}"
                                )
                                print(
                                    f"Rank {rank} norm_out ref value at max diff: {ref_norm_out.view(-1)[max_diff_idx]}"
                                )

                            torch.testing.assert_close(
                                norm_out.to(torch.float32),
                                ref_norm_out,
                                atol=tolerance,
                                rtol=1e-2,
                            )
                            # 6.4 Check quant_out
                            # todo

                        dist.barrier(group=group)
                        if test_passed:
                            print(
                                f"test RANK {rank}: token{token_num}-expert{active_expert_num}-tp{world_size}-{dtype}-layout{swizzled_layout_code}-pdl{launch_with_pdl} passed"
                            )
                        else:
                            print(
                                f"test RANK {rank}: token{token_num}-expert{active_expert_num}-tp{world_size}-{dtype}-layout{swizzled_layout_code}-pdl{launch_with_pdl} failed"
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
        assert procs[i].exitcode == 0, (
            f"Process {i} failed with exit code {procs[i].exitcode}"
        )


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_trtllm_moe_allreduce_fusion(world_size, dtype):
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
        _run_correctness_worker,
        target_args=(),
    )
    print(f"moe allreduce fusion tp = {world_size}: OK")


if __name__ == "__main__":
    test_trtllm_moe_allreduce_fusion(2, torch.float16)
