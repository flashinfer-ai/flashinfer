# Test for unified AllReduce API with multiple backends
# Run with: mpirun -np <num_gpus> pytest tests/comm/test_allreduce_unified_api.py -vv -s
import traceback
from typing import Tuple

import pytest
import torch
from mpi4py import MPI

import flashinfer.comm.trtllm_mnnvl_ar as trtllm_mnnvl_ar
from flashinfer.comm.mnnvl import TorchDistBackend

# Unified API imports
from flashinfer.comm import (
    create_allreduce_fusion_workspace,
    allreduce_fusion,
    AllReduceFusionPattern,
    AllReduceFusionWorkspace,
)

# Use flashinfer.norm.rmsnorm as reference implementation.
from flashinfer.norm import rmsnorm

# Test helpers
from tests.test_helpers.comm import (
    init_torch_distributed_from_mpi,
    cleanup_torch_distributed,
)


@torch.inference_mode()
def run_allreduce_fusion_test(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    rank: int,
    fusion: bool,
    reference_output: tuple[torch.Tensor, ...],
    workspace: AllReduceFusionWorkspace,
):
    """Test function using the unified API (create_allreduce_fusion_workspace + allreduce_fusion)."""
    MPI.COMM_WORLD.barrier()

    def func(
        input,
        residual,
        norm_weight,
        eps,
        enable_fusion,
        workspace,
    ):
        # For both fused and unfused cases:
        shape = input.shape
        input = input.view(-1, shape[-1])
        use_pdl = True

        if enable_fusion:
            trtllm_mnnvl_ar.mpi_barrier()

            # Use unified API
            norm_out = torch.empty_like(input)
            residual_out = torch.empty_like(input)

            allreduce_fusion(
                input=input,
                workspace=workspace,
                pattern=AllReduceFusionPattern.kARResidualRMSNorm,
                launch_with_pdl=use_pdl,
                residual_out=residual_out,
                norm_out=norm_out,
                residual_in=residual.view(-1, shape[-1]),
                rms_gamma=norm_weight,
                rms_eps=eps,
            )

            return norm_out.view(shape), residual_out.view(shape)

        else:
            # Use unified API for AllReduce only
            output = torch.empty_like(input)

            allreduce_fusion(
                input=input,
                workspace=workspace,
                pattern=AllReduceFusionPattern.kAllReduce,
                launch_with_pdl=use_pdl,
                output=output,
            )
            return (output.view(shape),)

    output = func(x.clone(), residual.clone(), norm_weight, eps, fusion, workspace)

    assert output[0].shape == reference_output[0].shape

    if rank == 0:
        print("output[0] (first 10 values):", output[0].flatten()[:10])
        print(
            "reference_output[0] (first 10 values):",
            reference_output[0].flatten()[:10],
        )

        if fusion:
            print("output[1] (first 10 values):", output[1].flatten()[:10])
            print(
                "reference_output[1] (first 10 values):",
                reference_output[1].flatten()[:10],
            )

    torch.testing.assert_close(
        output[0],
        reference_output[0],
        rtol=0.05,
        atol=0.15,
    )

    if fusion:
        torch.testing.assert_close(
            output[1],
            reference_output[1],
            rtol=0.05,
            atol=0.15,
        )


def prepare_test_data(seq_len: int, hidden_size: int, dtype: torch.dtype, fusion: bool):
    """Prepare test data distributed across MPI ranks."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    if rank == 0:
        x_full = torch.randn((world_size, seq_len, hidden_size), dtype=dtype)
        residual = torch.randn((seq_len, hidden_size), dtype=dtype)
        norm_weight = torch.randn((hidden_size,), dtype=dtype)
    else:
        x_full = None
        residual = None
        norm_weight = None

    # Use lowercase bcast() for Python object broadcasting
    x_full = comm.bcast(x_full, root=0)
    residual = comm.bcast(residual, root=0)
    norm_weight = comm.bcast(norm_weight, root=0)

    x_full = x_full.cuda()
    residual = residual.cuda()
    norm_weight = norm_weight.cuda()

    x_local = x_full[rank, :, :]
    reference_output: Tuple[torch.Tensor, ...] = None
    if fusion:
        # Fused case: AllReduce + Residual Add + RMS Norm
        allreduce_result = torch.sum(x_full, dim=0)  # AllReduce result
        residual_out = allreduce_result + residual  # Add residual
        norm_out = rmsnorm(
            residual_out, norm_weight, torch.finfo(dtype).eps, enable_pdl=False
        )

        reference_output = (norm_out, residual_out)
    else:
        # Non-fused case: Only AllReduce
        allreduce_result = torch.sum(x_full, dim=0)  # AllReduce result
        reference_output = (allreduce_result,)
    return (x_local, residual, norm_weight), reference_output


def run_allreduce_test(
    monkeypatch,
    seq_lens: list[int],
    fusion: bool,
    dtype: torch.dtype,
    hidden_size: int,
    backend: str,
):
    """Core test logic for AllReduce operations using the unified API.

    Args:
        monkeypatch: pytest monkeypatch fixture
        seq_lens: List of sequence lengths to test
        fusion: Whether to test fused allreduce+rmsnorm or just allreduce
        dtype: Data type for tensors
        hidden_size: Hidden dimension size
        backend: Backend to use ("auto", "trtllm", "mnnvl")
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    gpus_per_node = torch.cuda.device_count()

    if gpus_per_node == 0:
        pytest.skip("AllReduce test requires at least one CUDA device per node")
    if world_size < 2:
        pytest.skip(f"This test requires at least 2 MPI ranks, got {world_size}")

    # Set CUDA device based on rank
    local_rank = rank % gpus_per_node
    torch.cuda.set_device(local_rank)

    if local_rank == 0:
        print(f"Running AllReduce test with {world_size} ranks, backend={backend}")
        print(f"Rank {rank} using GPU {torch.cuda.current_device()}")

    eps = 1e-5
    torch.manual_seed(0)

    workspace = None

    try:
        init_torch_distributed_from_mpi()
        # Create workspace using unified API
        workspace = create_allreduce_fusion_workspace(
            backend=backend,
            world_size=world_size,
            rank=rank,
            max_token_num=max(seq_lens),
            hidden_dim=hidden_size,
            dtype=dtype,
            gpus_per_node=gpus_per_node,
            comm_backend=TorchDistBackend(),
        )

        print(f"Rank {rank}: Created workspace with backend={workspace.backend}")

        # Prepare test data for all sequence lengths
        test_data = []
        for seq_len in seq_lens:
            (x_local, residual, norm_weight), reference_output = prepare_test_data(
                seq_len, hidden_size, dtype, fusion
            )
            test_data.append(
                (seq_len, x_local, residual, norm_weight, reference_output)
            )

        # Test each sequence length with the same workspace
        for seq_len, x, residual, norm_weight, reference_output in test_data:
            if rank == 0:
                print(
                    f"Testing seq_len={seq_len}, hidden_size={hidden_size}, fusion={fusion}, dtype={dtype}"
                )

            run_allreduce_fusion_test(
                x,
                residual,
                norm_weight,
                eps,
                rank,
                fusion,
                reference_output,
                workspace,
            )

            # Synchronize before next test
            trtllm_mnnvl_ar.mpi_barrier()

            print(
                f"PASSED[rank={rank}]: seq_len={seq_len}, fusion={fusion}, dtype={dtype}, backend={backend}"
            )

    except Exception as e:
        failure_message = f"FAILED[rank={rank}]: seq_lens={seq_lens}, fusion={fusion}, dtype={dtype}, backend={backend} failed: {e}"
        print(failure_message)
        print(traceback.format_exc())

        # Gather failure status from all ranks for logging
        all_failures = MPI.COMM_WORLD.allgather(True)
        failed_ranks = [i for i, failed in enumerate(all_failures) if failed]
        if rank == 0:
            print(f"Test failed on ranks: {failed_ranks}")

        raise

    finally:
        if workspace is not None:
            workspace.destroy()
        # Cleanup torch.distributed if we initialized it
        if backend in ("trtllm", "auto"):
            cleanup_torch_distributed()

    # Final synchronization
    trtllm_mnnvl_ar.mpi_barrier()


@pytest.mark.parametrize(
    "seq_lens",
    [[1], [4], [15], [27, 11, 24, 256], [127], [998, 2048]],
)
@pytest.mark.parametrize("fusion", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [2880, 7168])
@pytest.mark.parametrize("backend", ["auto", "trtllm", "mnnvl"])
def test_allreduce_unified(
    monkeypatch,
    seq_lens: list[int],
    fusion: bool,
    dtype: torch.dtype,
    hidden_size: int,
    backend: str,
):
    """Test AllReduce with unified API across different backends.

    Run with: mpirun -np <num_gpus> pytest tests/comm/test_allreduce_unified_api.py -vv -s
    """
    run_allreduce_test(monkeypatch, seq_lens, fusion, dtype, hidden_size, backend)
