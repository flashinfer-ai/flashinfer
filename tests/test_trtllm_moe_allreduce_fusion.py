import multiprocessing as mp
import socket
from typing import Any

import pytest
import torch
import torch.distributed as dist
import numpy as np

import flashinfer.comm as comm

# todo(Yingyi): add warmup stes for benchmark

# Usage: test var, temp
kOneShotMaxTokenNum = 128
MAX_TOKEN_NUM = 2048
# MAX_TOKEN_NUM = kOneShotMaxTokenNum
# HIDDEN_SIZE = 7168
HIDDEN_SIZE = 16  # debug-only
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
            # 1,
            # 64,
            128,
            # 256,
            # 2048,
        ]  # 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
        # candidate_active_expert_num = [8, 12, 16]
        candidate_active_expert_num = [1]
        fusion_codes = [
            MoEAllReduceFusionType.RESIDUAL_QUANT_OUT,
            # MoEAllReduceFusionType.NORM_OUT,
            # MoEAllReduceFusionType.RESIDUAL_NORM_OUT,
            # MoEAllReduceFusionType.RESIDUAL_NORM_QUANT_OUT,
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

        # ipc_handles, workspace_tensor = (
        #     comm.trtllm_create_ipc_workspace_for_all_reduce_fusion_debug(
        #         rank, world_size, MAX_TOKEN_NUM, HIDDEN_SIZE, group=group
        #     )
        # )

        test_loop = 1  # make it 5 after debug

        for token_num in token_nums:
            for active_expert_num in candidate_active_expert_num:
                for fusion_code in fusion_codes:
                    for swizzled_layout_code in swizzled_layout_codes:
                        for launch_with_pdl in launch_with_pdls:
                            test_passed = True
                            # print(
                            #     f"test RANK {rank}: token{token_num}-expert{active_expert_num}-tp{world_size}-{dtype}-fusion{fusion_code}-layout{swizzled_layout_code}-pdl{launch_with_pdl} start"
                            # )
                            dist.barrier(group=group)
                            torch.cuda.synchronize()
                            for _ in range(test_loop):
                                message_size = token_num * HIDDEN_SIZE

                                inp1 = torch.randn(
                                    message_size, dtype=dtype, device=device
                                )

                                # init params for each fusion op
                                residual_in = torch.randn(
                                    message_size, dtype=dtype, device=device
                                )
                                residual_out = torch.empty_like(residual_in)
                                norm_out = torch.empty_like(residual_in)
                                quant_out = torch.empty(
                                    message_size // 2, dtype=torch.uint8, device=device
                                )
                                scale_out = torch.empty(
                                    message_size // 16, dtype=torch.uint8, device=device
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

                                # init moe params
                                # [device_num_expert, m]
                                # moe_reduction_scale_input = torch.randn(
                                #     active_expert_num * token_num,
                                #     dtype=torch.float32,
                                #     device=device,
                                # )

                                # debug-only
                                # set moe_reduction_scale_input to 1.0
                                moe_reduction_scale_input = torch.ones(
                                    active_expert_num * token_num,
                                    dtype=torch.float32,
                                    device=device,
                                )
                                moe_reduction_scale_input_clone = moe_reduction_scale_input.clone()

                                # [device_num_expert, m, 7168]
                                moe_reduction_active_experts_token_input = torch.randn(
                                    active_expert_num * message_size,
                                    dtype=dtype,
                                    device=device,
                                )
                                moe_reduction_active_experts_token_input_clone = moe_reduction_active_experts_token_input.clone()
                                # [m, 7168]
                                moe_reduction_token_input = torch.randn(
                                    message_size, dtype=dtype, device=device
                                )
                                moe_reduction_token_input_clone = moe_reduction_token_input.clone()

                                # debug-only
                                # print first 10 elements of moe_reduction_active_experts_token_input and moe_reduction_token_input
                                print(f"moe_reduction_active_experts_token_input: {moe_reduction_active_experts_token_input}")
                                print(f"moe_reduction_token_input: {moe_reduction_token_input}")

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

                                torch.cuda.synchronize()

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
                                moe_out_ref = reduced_expert_out.view(
                                    message_size
                                ) + moe_reduction_token_input_clone.to(
                                    torch.float32
                                )  # [message_size]

                                # 3. All-Reduce
                                all_reduced_ref = moe_out_ref.clone().to(dtype)
                                dist.all_reduce(all_reduced_ref, group=group)
                                all_reduced_ref = all_reduced_ref.to(torch.float32)

                                # 4. Fused Ops
                                ref_residual_out = all_reduced_ref.view(
                                    token_num, HIDDEN_SIZE
                                ) + residual_in.view(token_num, HIDDEN_SIZE).to(
                                    torch.float32
                                )

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

                                # 5. Check correctness
                                # match shape
                                residual_out = residual_out.view(token_num, HIDDEN_SIZE)
                                norm_out = norm_out.view(token_num, HIDDEN_SIZE)

                                # fusion code 0: residual+quant
                                if (
                                    fusion_code
                                    == MoEAllReduceFusionType.RESIDUAL_QUANT_OUT
                                ):
                                    # if not torch.allclose(
                                    #     residual_out.to(torch.float32),
                                    #     ref_residual_out,
                                    #     atol=1e-2,
                                    #     rtol=1e-2,
                                    # ):
                                    #     test_passed = False
                                    #     print(f"Rank {rank} residual_out mismatch")
                                    #     print(f"residual_out: {residual_out}")
                                    #     print(f"ref_residual_out: {ref_residual_out}")
                                    
                                    # if not torch.allclose(
                                    #     residual_out.to(torch.float32),
                                    #     all_reduced_ref.view(token_num, HIDDEN_SIZE),
                                    #     atol=1e-2,
                                    #     rtol=1e-2,
                                    # ):
                                    if rank == 0:
                                        all_reduced_ref = all_reduced_ref.view(token_num, HIDDEN_SIZE)
                                        test_passed = False
                                        print(f"Rank {rank} all_reduce_out mismatch")
                                        print(f"all_reduce_out: {residual_out}")
                                        print(f"ref_all_reduce_out: {all_reduced_ref}")

                                    # # TODO(yingyi): add support for swizzled layout
                                    # if (
                                    #     swizzled_layout_code
                                    #     == comm.FP4QuantizationSFLayout.LINEAR
                                    # ):
                                    #     num_scales = token_num * HIDDEN_SIZE // 16
                                    #     dequant_scales = dequantize_fp8_e4m3_tensor(
                                    #         scale_out[:num_scales]
                                    #     )
                                    #     dequant_data = dequantize_fp4_e2m1_tensor(
                                    #         quant_out.view(torch.int32)[
                                    #             : num_scales * 2
                                    #         ]
                                    #     )
                                    #     dequant_out = (
                                    #         dequant_data.view(token_num, HIDDEN_SIZE)
                                    #         * dequant_scales.view(
                                    #             token_num, -1
                                    #         ).repeat_interleave(16, dim=1)
                                    #         / scale_factor.item()
                                    #     )
                                    #     if not torch.allclose(
                                    #         dequant_out,
                                    #         ref_norm_out,
                                    #         atol=1e-2,
                                    #         rtol=1e-2,
                                    #     ):
                                    #         print(
                                    #             f"Rank {rank}: Quantization check mismatch"
                                    #         )
                                    #         print(f"dequant_out: {dequant_out}")
                                    #         print(f"ref_norm_out: {ref_norm_out}")

                                # fusion code 1: norm
                                elif fusion_code == MoEAllReduceFusionType.NORM_OUT:
                                    if not torch.allclose(
                                        norm_out.to(torch.float32),
                                        ref_norm_out,
                                        atol=1e-2,
                                        rtol=1e-2,
                                    ):
                                        test_passed = False
                                        print(f"Rank {rank} norm_out mismatch")
                                        print(f"norm_out: {norm_out}")
                                        print(f"ref_norm_out: {ref_norm_out}")

                                    # # TODO(yingyi): add support for swizzled layout
                                    # if (
                                    #     swizzled_layout_code
                                    #     == comm.FP4QuantizationSFLayout.LINEAR
                                    # ):
                                    #     num_scales = token_num * HIDDEN_SIZE // 16
                                    #     dequant_scales = dequantize_fp8_e4m3_tensor(
                                    #         scale_out[:num_scales]
                                    #     )
                                    #     dequant_data = dequantize_fp4_e2m1_tensor(
                                    #         quant_out.view(torch.int32)[
                                    #             : num_scales * 2
                                    #         ]
                                    #     )
                                    #     dequant_out = (
                                    #         dequant_data.view(token_num, HIDDEN_SIZE)
                                    #         * dequant_scales.view(
                                    #             token_num, -1
                                    #         ).repeat_interleave(16, dim=1)
                                    #         / scale_factor.item()
                                    #     )
                                    #     if not torch.allclose(
                                    #         dequant_out,
                                    #         ref_norm_out,
                                    #         atol=1e-2,
                                    #         rtol=1e-2,
                                    #     ):
                                    #         print(f"Rank {rank}: Quantization check mismatch")
                                    #         print(f"dequant_out: {dequant_out}")
                                    #         print(f"ref_norm_out: {ref_norm_out}")

                                # fusion code 2: residual+norm
                                elif (
                                    fusion_code
                                    == MoEAllReduceFusionType.RESIDUAL_NORM_OUT
                                ):
                                    if not torch.allclose(
                                        residual_out.to(torch.float32),
                                        ref_residual_out,
                                        atol=1e-2,
                                        rtol=1e-2,
                                    ):
                                        test_passed = False
                                        print(f"Rank {rank} residual_out mismatch")
                                        print(f"residual_out: {residual_out}")
                                        print(f"ref_residual_out: {ref_residual_out}")

                                    if not torch.allclose(
                                        norm_out.to(torch.float32),
                                        ref_norm_out,
                                        atol=1e-2,
                                        rtol=1e-2,
                                    ):
                                        test_passed = False
                                        print(f"Rank {rank} norm_out mismatch")
                                # fusion code 3: residual+norm+quant
                                elif (
                                    fusion_code
                                    == MoEAllReduceFusionType.RESIDUAL_NORM_QUANT_OUT
                                ):
                                    if not torch.allclose(
                                        residual_out.to(torch.float32),
                                        ref_residual_out,
                                        atol=1e-2,
                                        rtol=1e-2,
                                    ):
                                        test_passed = False
                                        print(f"Rank {rank} residual_out mismatch")
                                        print(f"residual_out: {residual_out}")
                                        print(f"ref_residual_out: {ref_residual_out}")

                                    if not torch.allclose(
                                        norm_out.to(torch.float32),
                                        ref_norm_out,
                                        atol=1e-2,
                                        rtol=1e-2,
                                    ):
                                        test_passed = False
                                        print(f"Rank {rank} norm_out mismatch")
                                        print(f"norm_out: {norm_out}")
                                        print(f"ref_norm_out: {ref_norm_out}")

                                    # # TODO(yingyi): add support for swizzled layout
                                    # if (
                                    #     swizzled_layout_code
                                    #     == comm.FP4QuantizationSFLayout.LINEAR
                                    # ):
                                    #     num_scales = token_num * HIDDEN_SIZE // 16
                                    #     dequant_scales = dequantize_fp8_e4m3_tensor(
                                    #         scale_out[:num_scales]
                                    #     )
                                    #     dequant_data = dequantize_fp4_e2m1_tensor(
                                    #         quant_out.view(torch.int32)[
                                    #             : num_scales * 2
                                    #         ]
                                    #     )
                                    #     dequant_out = (
                                    #         dequant_data.view(token_num, HIDDEN_SIZE)
                                    #         * dequant_scales.view(
                                    #             token_num, -1
                                    #         ).repeat_interleave(16, dim=1)
                                    #         / scale_factor.item()
                                    #     )
                                    #     if not torch.allclose(
                                    #         dequant_out,
                                    #         ref_norm_out,
                                    #         atol=1e-2,
                                    #         rtol=1e-2,
                                    #     ):
                                    #         print(
                                    #             f"Rank {rank}: Quantization check mismatch"
                                    #         )
                                    #         print(f"dequant_out: {dequant_out}")
                                    #         print(f"ref_norm_out: {ref_norm_out}")

                            dist.barrier(group=group)
                            if test_passed:
                                print(
                                    f"test RANK {rank}: token{token_num}-expert{active_expert_num}-tp{world_size}-{dtype}-fusion{fusion_code}-layout{swizzled_layout_code}-pdl{launch_with_pdl} passed"
                                )
                            else:
                                print(
                                    f"test RANK {rank}: token{token_num}-expert{active_expert_num}-tp{world_size}-{dtype}-fusion{fusion_code}-layout{swizzled_layout_code}-pdl{launch_with_pdl} failed"
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
    torch.manual_seed(42)
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


def dequantize_fp8_e4m3_tensor(scale_tensor_uint8: torch.Tensor) -> torch.Tensor:
    """Dequantize a tensor of FP8 E4M3 values."""
    sign = ((scale_tensor_uint8 >> 7) & 1).float()
    exp = ((scale_tensor_uint8 >> 3) & 0b1111).float()
    mant = (scale_tensor_uint8 & 0b111).float()

    val = torch.zeros_like(scale_tensor_uint8, dtype=torch.float32)

    # Handle subnormal numbers
    is_subnormal = exp == 0
    # Handle normal numbers
    is_normal = (exp > 0) & (exp < 15)

    # Normal numbers (bias 7)
    val[is_normal] = (
        ((-1.0) ** sign[is_normal])
        * (2.0 ** (exp[is_normal] - 7.0))
        * (1.0 + mant[is_normal] / 8.0)
    )
    # Subnormal numbers (bias 7)
    # value = (-1)^S * 2^(1-bias) * (0.M) = (-1)^S * 2^(-6) * (M/8)
    val[is_subnormal] = (
        ((-1.0) ** sign[is_subnormal]) * (2.0**-6.0) * (mant[is_subnormal] / 8.0)
    )
    # Inf/NaN cases (exp=15) will be dequantized to 0, which is a reasonable default.
    return val


def dequantize_fp4_e2m1_tensor(quant_tensor_int32: torch.Tensor) -> torch.Tensor:
    """Dequantize a tensor of FP4 E2M1 values packed in int32."""
    shape = quant_tensor_int32.shape
    # PyTorch does not support uint32, so we use int32 and view it as uint8
    quant_tensor_uint8 = quant_tensor_int32.view(torch.uint8)

    # Unpack each uint32 into four uint8
    unpacked_bytes = quant_tensor_uint8.view(shape[:-1] + (-1, 4))
    b0 = unpacked_bytes[..., 0]
    b1 = unpacked_bytes[..., 1]
    b2 = unpacked_bytes[..., 2]
    b3 = unpacked_bytes[..., 3]

    # Unpack each uint8 into two 4-bit nibbles
    fp4_0 = b0 & 0x0F
    fp4_1 = b0 >> 4
    fp4_2 = b1 & 0x0F
    fp4_3 = b1 >> 4
    fp4_4 = b2 & 0x0F
    fp4_5 = b2 >> 4
    fp4_6 = b3 & 0x0F
    fp4_7 = b3 >> 4

    # Stack and transpose to get (..., 8) shape
    fp4_vals = torch.stack(
        [fp4_0, fp4_1, fp4_2, fp4_3, fp4_4, fp4_5, fp4_6, fp4_7], dim=-1
    )

    sign = ((fp4_vals >> 3) & 1).float()
    exp = ((fp4_vals >> 1) & 0b11).float()
    mant = (fp4_vals & 0b1).float()

    val = torch.zeros_like(fp4_vals, dtype=torch.float32)

    is_subnormal = exp == 0
    # The kernel comment notes a max value of 6.0, which implies that the
    # maximum exponent (3) is used for normal numbers, not Inf/NaN.
    is_normal = exp > 0

    # Normal numbers (bias 1)
    val[is_normal] = (
        ((-1.0) ** sign[is_normal])
        * (2.0 ** (exp[is_normal] - 1.0))
        * (1.0 + mant[is_normal] / 2.0)
    )

    # Subnormal numbers (bias 1)
    # value = (-1)^S * 2^(1-bias) * (0.M)
    val[is_subnormal] = ((-1.0) ** sign[is_subnormal]) * (mant[is_subnormal] / 2.0)

    return val.view(shape[:-1] + (-1,))


if __name__ == "__main__":
    mod = comm.get_comm_module()
    # set random seed
    # torch.manual_seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    test_trtllm_moe_allreduce_fusion(2, torch.float16)
    # test_trtllm_moe_allreduce_fusion(2, torch.bfloat16)
    # test_trtllm_moe_allreduce_fusion(4, torch.float16)
    # test_trtllm_moe_allreduce_fusion(4, torch.bfloat16)
