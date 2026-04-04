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

import flashinfer.comm as comm
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
    distributed_init_method = f"tcp://localhost:{distributed_init_port}"
    dist.init_process_group(
        backend="nccl",
        init_method=distributed_init_method,
        rank=rank,
        world_size=world_size,
    )
    group = dist.group.WORLD

    try:
        seq_len = residual.shape[0]
        top_k = expanded_idx_to_permuted_idx.shape[1]
        eps = 1e-5

        # Create unified workspace
        workspace = TRTLLMAllReduceFusionWorkspace(
            tp_size=world_size,
            tp_rank=rank,
            max_token_num=MAX_TOKEN_NUM,
            hidden_dim=HIDDEN_SIZE,
            dtype=dtype,
        )

        # Move tensors to device
        shared_expert_output_d = shared_expert_output.to(device)
        fc2_output_d = fc2_output.to(device)
        scale_d = scale.to(device)
        expanded_idx_d = expanded_idx_to_permuted_idx.to(device)
        residual_d = residual.to(device)
        norm_weight = torch.randn((HIDDEN_SIZE,), dtype=dtype, device=device)

        norm_out = torch.empty_like(residual_d)
        residual_out = torch.empty_like(residual_d)

        dist.barrier(group=group)

        # Call via unified API
        result = allreduce_fusion(
            input=fc2_output_d,
            workspace=workspace,
            pattern=AllReduceFusionPattern.kMoEFinalizeARResidualRMSNorm,
            launch_with_pdl=False,
            residual_in=residual_d,
            residual_out=residual_out,
            norm_out=norm_out,
            rms_gamma=norm_weight,
            rms_eps=eps,
            expanded_idx_to_permuted_idx=expanded_idx_d,
            expert_scale_factor=scale_d,
            shared_expert_output=shared_expert_output_d,
        )

        torch.cuda.synchronize()

        # Reference computation
        fc2_output_cpu = fc2_output.to(torch.float32)
        expert_reduction = torch.sum(
            fc2_output_cpu[expanded_idx_to_permuted_idx] * scale.unsqueeze(-1).float(),
            dim=1,
        )
        torch_before_residual = (
            expert_reduction + shared_expert_output.float()
        ) * world_size
        torch_residual = torch_before_residual + residual.float()
        torch_norm = _rms_norm(torch_residual, norm_weight.cpu().float(), eps).to(dtype)

        # Check
        torch.testing.assert_close(
            residual_out.cpu().to(torch.float32),
            torch_residual.to(torch.float32),
            rtol=0.2,
            atol=0.2,
        )
        torch.testing.assert_close(
            norm_out.cpu().to(torch.float32),
            torch_norm.cpu().to(torch.float32),
            rtol=0.2,
            atol=0.2,
        )

        # Verify return value is norm_out
        assert result.data_ptr() == norm_out.data_ptr()

        dist.barrier(group=group)
        if rank == 0:
            print(
                f"MOE Finalize unified API: tp{world_size}-{dtype} PASSED"
            )

    finally:
        dist.barrier(group=group)
        comm.trtllm_destroy_ipc_workspace_for_all_reduce(
            workspace.ipc_handles, group=group
        )
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
def test_moe_finalize_allreduce_unified_api(world_size, dtype):
    """Test kMoEFinalizeARResidualRMSNorm pattern via allreduce_fusion()."""
    available_gpus = torch.cuda.device_count()
    if world_size > available_gpus:
        pytest.skip(
            f"world_size {world_size} > available_gpus {available_gpus}"
        )

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    seq_len = 16
    top_k = 8

    shared_expert_output = torch.randn((seq_len, HIDDEN_SIZE), dtype=dtype)
    fc2_output = torch.randn((seq_len * top_k, HIDDEN_SIZE), dtype=dtype)
    scale = torch.randn((seq_len, top_k), dtype=dtype)
    expanded_idx_to_permuted_idx = torch.randint(
        0, seq_len * top_k, (seq_len, top_k), dtype=torch.int32
    )
    residual = torch.randn_like(shared_expert_output)

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
