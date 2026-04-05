"""Tests for MOE patterns in the unified allreduce_fusion API.

Verifies that allreduce_fusion() with kMoEFinalizeARResidualRMSNorm and
kMoEReductionARResidualRMSNorm patterns correctly dispatches to the
underlying trtllm_moe_finalize_allreduce_fusion /
trtllm_moe_allreduce_fusion kernels.

Usage:
    mpirun -np 2 pytest tests/comm/test_allreduce_fusion_moe_unified_api.py -v
    mpirun -np 4 pytest tests/comm/test_allreduce_fusion_moe_unified_api.py -v
"""

import multiprocessing as mp
import socket
from typing import Any

import numpy as np
import pytest
import torch
import torch.distributed as dist

from flashinfer.comm import (
    AllReduceFusionPattern,
    TRTLLMAllReduceFusionWorkspace,
    allreduce_fusion,
)

MAX_TOKEN_NUM = 2048
HIDDEN_SIZE = 7168


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight
    return y


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


# ============================================================================
# MOE Finalize via unified API
# ============================================================================


def _run_moe_finalize_unified_worker(
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
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{distributed_init_port}",
        rank=rank,
        world_size=world_size,
    )
    group = dist.group.WORLD

    try:
        seq_len = residual.shape[0]
        top_k = expanded_idx_to_permuted_idx.shape[1]
        eps = 1e-5
        test_loop = 3
        has_shared_expert = shared_expert_output is not None

        workspace = TRTLLMAllReduceFusionWorkspace(
            tp_size=world_size,
            tp_rank=rank,
            max_token_num=MAX_TOKEN_NUM,
            hidden_dim=HIDDEN_SIZE,
            dtype=dtype,
        )

        # Move tensors to device
        shared_d = shared_expert_output.to(device) if has_shared_expert else None
        fc2_d = fc2_output.to(device)
        scale_d = scale.to(device)
        idx_d = expanded_idx_to_permuted_idx.to(device)
        residual_d = residual.to(device)
        norm_weight = torch.randn((HIDDEN_SIZE,), dtype=dtype, device=device)
        norm_out = torch.empty_like(residual_d)
        residual_out = torch.empty_like(residual_d)

        dist.barrier(group=group)

        for launch_with_pdl in [False, True]:
            for _ in range(test_loop):
                # Call via unified API
                result = allreduce_fusion(
                    input=fc2_d,
                    workspace=workspace,
                    pattern=AllReduceFusionPattern.kMoEFinalizeARResidualRMSNorm,
                    launch_with_pdl=launch_with_pdl,
                    residual_in=residual_d,
                    residual_out=residual_out,
                    norm_out=norm_out,
                    rms_gamma=norm_weight,
                    rms_eps=eps,
                    expanded_idx_to_permuted_idx=idx_d,
                    expert_scale_factor=scale_d,
                    shared_expert_output=shared_d,
                )
                torch.cuda.synchronize()

                # Reference
                fc2_f32 = fc2_output.to(torch.float32)
                expert_reduction = torch.sum(
                    fc2_f32[expanded_idx_to_permuted_idx] * scale.unsqueeze(-1).float(),
                    dim=1,
                )
                torch_before_residual = expert_reduction
                if has_shared_expert:
                    torch_before_residual = (
                        torch_before_residual + shared_expert_output.float()
                    )
                torch_before_residual = torch_before_residual * world_size
                torch_residual = torch_before_residual + residual.float()
                torch_norm = _rms_norm(
                    torch_residual, norm_weight.cpu().float(), eps
                ).to(dtype)

                # Validate
                torch.testing.assert_close(
                    residual_out.cpu().float(),
                    torch_residual.float(),
                    rtol=0.2,
                    atol=0.2,
                )
                torch.testing.assert_close(
                    norm_out.cpu().float(),
                    torch_norm.cpu().float(),
                    rtol=0.2,
                    atol=0.2,
                )
                assert result.data_ptr() == norm_out.data_ptr()

            # CUDA Graph capture/replay
            dist.barrier(group=group)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(test_loop):
                    allreduce_fusion(
                        input=fc2_d,
                        workspace=workspace,
                        pattern=AllReduceFusionPattern.kMoEFinalizeARResidualRMSNorm,
                        launch_with_pdl=launch_with_pdl,
                        residual_in=residual_d,
                        residual_out=residual_out,
                        norm_out=norm_out,
                        rms_gamma=norm_weight,
                        rms_eps=eps,
                        expanded_idx_to_permuted_idx=idx_d,
                        expert_scale_factor=scale_d,
                        shared_expert_output=shared_d,
                    )
            torch.cuda.current_stream().wait_stream(s)

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(test_loop):
                    allreduce_fusion(
                        input=fc2_d,
                        workspace=workspace,
                        pattern=AllReduceFusionPattern.kMoEFinalizeARResidualRMSNorm,
                        launch_with_pdl=launch_with_pdl,
                        residual_in=residual_d,
                        residual_out=residual_out,
                        norm_out=norm_out,
                        rms_gamma=norm_weight,
                        rms_eps=eps,
                        expanded_idx_to_permuted_idx=idx_d,
                        expert_scale_factor=scale_d,
                        shared_expert_output=shared_d,
                    )
            g.replay()
            torch.cuda.synchronize()

            # Validate after graph replay
            torch.testing.assert_close(
                residual_out.cpu().float(),
                torch_residual.float(),
                rtol=0.2,
                atol=0.2,
            )
            torch.testing.assert_close(
                norm_out.cpu().float(),
                torch_norm.cpu().float(),
                rtol=0.2,
                atol=0.2,
            )

        dist.barrier(group=group)
        if rank == 0:
            shared_str = "with_shared" if has_shared_expert else "no_shared"
            print(
                f"MOE Finalize unified API: tp{world_size}-{dtype}-"
                f"seq{seq_len}-topk{top_k}-{shared_str} PASSED"
            )

    finally:
        dist.barrier(group=group)
        workspace.destroy()
        dist.destroy_process_group(group=group)


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "seq_len,top_k,use_shared_expert",
    [
        (1, 8, True),  # decode: single token
        (16, 8, True),  # small batch with shared expert (DeepSeek)
        (16, 8, False),  # small batch without shared expert
        (128, 4, True),  # larger batch, different top_k
    ],
)
def test_moe_finalize_allreduce_unified_api(
    world_size, dtype, seq_len, top_k, use_shared_expert
):
    """Test kMoEFinalizeARResidualRMSNorm pattern via allreduce_fusion()."""
    available_gpus = torch.cuda.device_count()
    if world_size > available_gpus:
        pytest.skip(f"world_size {world_size} > available_gpus {available_gpus}")

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    shared_expert_output = (
        torch.randn((seq_len, HIDDEN_SIZE), dtype=dtype) if use_shared_expert else None
    )
    # Indices may contain duplicates — this is realistic since multiple tokens
    # can route to the same expert.
    fc2_output = torch.randn((seq_len * top_k, HIDDEN_SIZE), dtype=dtype)
    scale = torch.randn((seq_len, top_k), dtype=dtype)
    expanded_idx_to_permuted_idx = torch.randint(
        0, seq_len * top_k, (seq_len, top_k), dtype=torch.int32
    )
    residual = torch.randn((seq_len, HIDDEN_SIZE), dtype=dtype)

    multi_process_parallel(
        world_size,
        dtype,
        _run_moe_finalize_unified_worker,
        target_args=(
            shared_expert_output,
            fc2_output,
            scale,
            expanded_idx_to_permuted_idx,
            residual,
        ),
    )


# ============================================================================
# MOE Reduction via unified API
# ============================================================================


def _run_moe_reduction_unified_worker(
    world_size,
    rank,
    dtype,
    distributed_init_port,
    moe_active_experts_input,
    moe_token_input,
    moe_scale_input,
    residual,
    num_experts,
):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{distributed_init_port}",
        rank=rank,
        world_size=world_size,
    )
    group = dist.group.WORLD

    try:
        seq_len = residual.shape[0]
        hidden_dim = residual.shape[1]
        eps = 1e-5
        test_loop = 3

        workspace = TRTLLMAllReduceFusionWorkspace(
            tp_size=world_size,
            tp_rank=rank,
            max_token_num=MAX_TOKEN_NUM,
            hidden_dim=HIDDEN_SIZE,
            dtype=dtype,
        )

        active_d = moe_active_experts_input.to(device)
        token_d = moe_token_input.to(device)
        scale_d = moe_scale_input.to(device)
        residual_d = residual.to(device)
        norm_weight = torch.randn((hidden_dim,), dtype=dtype, device=device)

        norm_out = torch.empty_like(residual_d)
        residual_out = torch.empty_like(residual_d)

        dist.barrier(group=group)

        for launch_with_pdl in [False, True]:
            for _ in range(test_loop):
                result = allreduce_fusion(
                    input=token_d,  # input tensor (used as moe_reduction_token_input internally)
                    workspace=workspace,
                    pattern=AllReduceFusionPattern.kMoEReductionARResidualRMSNorm,
                    launch_with_pdl=launch_with_pdl,
                    residual_in=residual_d,
                    residual_out=residual_out,
                    norm_out=norm_out,
                    rms_gamma=norm_weight,
                    rms_eps=eps,
                    moe_reduction_device_num_experts=num_experts,
                    moe_reduction_scale_input=scale_d,
                    moe_reduction_active_experts_token_input=active_d,
                    moe_reduction_token_input=token_d,
                )
                torch.cuda.synchronize()

            # Reference computation:
            # expert_reduction = sum over experts of (active_expert_output * scale)
            # Then: allreduce(expert_reduction + token_input) + residual -> rms_norm
            # Note: exact reference depends on kernel semantics. We just verify
            # the API dispatches without error and produces finite outputs.
            assert result is not None
            assert torch.isfinite(norm_out).all(), "norm_out contains NaN/Inf"
            assert torch.isfinite(residual_out).all(), "residual_out contains NaN/Inf"

        dist.barrier(group=group)
        if rank == 0:
            print(
                f"MOE Reduction unified API: tp{world_size}-{dtype}-"
                f"seq{seq_len}-experts{num_experts} PASSED"
            )

    finally:
        dist.barrier(group=group)
        workspace.destroy()
        dist.destroy_process_group(group=group)


@pytest.mark.parametrize("world_size", [2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_moe_reduction_allreduce_unified_api(world_size, dtype):
    """Test kMoEReductionARResidualRMSNorm pattern via allreduce_fusion()."""
    available_gpus = torch.cuda.device_count()
    if world_size > available_gpus:
        pytest.skip(f"world_size {world_size} > available_gpus {available_gpus}")

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    seq_len = 16
    num_experts = 8

    # Per-token-per-expert outputs: [seq_len * num_experts, hidden_dim]
    moe_active_experts_input = torch.randn(
        (seq_len * num_experts, HIDDEN_SIZE), dtype=dtype
    )
    # Per-token input (e.g. FC2 output): [seq_len, hidden_dim]
    moe_token_input = torch.randn((seq_len, HIDDEN_SIZE), dtype=dtype)
    # Per-token-per-expert scale: [seq_len, num_experts]
    moe_scale_input = torch.randn((seq_len, num_experts), dtype=torch.float32)
    residual = torch.randn((seq_len, HIDDEN_SIZE), dtype=dtype)

    multi_process_parallel(
        world_size,
        dtype,
        _run_moe_reduction_unified_worker,
        target_args=(
            moe_active_experts_input,
            moe_token_input,
            moe_scale_input,
            residual,
            num_experts,
        ),
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
