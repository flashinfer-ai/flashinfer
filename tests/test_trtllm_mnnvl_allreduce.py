# Check torch version:
import os
import sys
import traceback

import pytest
import torch
from mpi4py import MPI  # Added MPI import

import flashinfer.comm as comm
import flashinfer.comm.trtllm_mnnvl_ar as trtllm_mnnvl_ar
from flashinfer.comm.mapping import Mapping
from flashinfer.comm.mnnvl import McastDeviceMemory, McastGPUBuffer

# Use flashinfer.norm.rmsnorm as reference implementation.
from flashinfer.norm import rmsnorm


@torch.inference_mode()
def row_linear_residual_norm_fusion_forward(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hidden_size: int,
    dtype: torch.dtype,
    tensor_parallel_size: int,
    tensor_parallel_rank: int,
    fusion: bool,
    reference_output: tuple[torch.Tensor, ...],
):

    x = x.cuda()
    residual = residual.cuda()
    norm_weight = norm_weight.cuda()
    reference_output = tuple(t.cuda() for t in reference_output)

    MPI.COMM_WORLD.barrier()

    mapping = Mapping(
        world_size=tensor_parallel_size,
        tp_size=tensor_parallel_size,
        rank=tensor_parallel_rank,
    )

    def func(
        input,
        residual,
        norm_weight,
        eps,
        enable_fusion,
        multicast_ptr,
        buffer_ptrs_dev,
        unicast_ptr,
        max_num_elements_mnnvl,
    ):
        # For both fused and unfused cases:
        shape = input.shape

        hidden_size = shape[-1]

        assert max_num_elements_mnnvl % hidden_size == 0

        input = input.view(-1, shape[-1])

        buffer_M = max_num_elements_mnnvl // hidden_size

        if enable_fusion:
            use_pdl = True

            prenorm_output = torch.empty_like(residual)
            normed_output = torch.empty_like(residual)

            trtllm_mnnvl_ar.mpi_barrier()

            trtllm_mnnvl_ar.trtllm_mnnvl_fused_allreduce_rmsnorm(
                prenorm_output,
                normed_output,
                input,
                multicast_ptr,
                buffer_ptrs_dev,
                unicast_ptr,
                buffer_M,
                buffer_flags_mnnvl,
                tensor_parallel_size,
                tensor_parallel_rank,
                norm_weight,
                eps,
                residual,
                use_pdl,
            )

            return normed_output.view(shape), prenorm_output.view(shape)

        else:
            output = torch.empty_like(input)

            trtllm_mnnvl_ar.trtllm_mnnvl_all_reduce(
                input,
                multicast_ptr,
                buffer_ptrs_dev,
                buffer_M,
                buffer_flags_mnnvl,
                tensor_parallel_size,
                tensor_parallel_rank,
                True,  # wait_for_results
                False,  # launch_with_pdl
                output,  # Need to provide output tensor since we are writing them out.
            )
            return (output.view(shape),)

    # Get workspace buffers using MPI rank
    mcast_buffer_mnnvl, buffer_flags_mnnvl, max_num_elements_mnnvl = (
        trtllm_mnnvl_ar.get_allreduce_mnnvl_workspace(mapping, dtype)
    )

    multicast_ptr = mcast_buffer_mnnvl.get_multicast_ptr_as_int64()
    buffer_ptrs_dev = mcast_buffer_mnnvl.get_buffer_ptrs_dev_as_ctypes_ptr()
    unicast_ptr = mcast_buffer_mnnvl.mcast_device_memory.get_unicast_ptr(
        tensor_parallel_rank
    )

    try:
        output = func(
            x.clone(),
            residual.clone(),
            norm_weight,
            eps,
            fusion,
            multicast_ptr,
            buffer_ptrs_dev,
            unicast_ptr,
            max_num_elements_mnnvl,
        )

        assert output[0].shape == reference_output[0].shape

        if tensor_parallel_rank == 0:
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

    finally:
        # Ensure cleanup happens even if assertions fail
        del mcast_buffer_mnnvl


"""Main test function that runs on each MPI rank"""


# seq_lens = [1, 4, 32, 128]
@pytest.mark.parametrize("seq_len", [4])
@pytest.mark.parametrize("fusion", [False, True])
def test_mnnvl_allreduce_full(monkeypatch, seq_len: int, fusion: bool):
    monkeypatch.setenv("TRTLLM_FORCE_MNNVL_AR", "1")  # force multi-node allreduce.

    # Get MPI info
    rank = MPI.COMM_WORLD.Get_rank()
    world_size = MPI.COMM_WORLD.Get_size()

    # Ensure we have exactly 2 ranks for this test
    if world_size < 2:
        if rank == 0:
            print(f"ERROR: This test requires at least 2 MPI ranks, got {world_size}")
        sys.exit(1)

    # Set CUDA device based on rank
    torch.cuda.set_device(rank)

    if rank == 0:
        print(f"Running MNNVL AllReduce test with {world_size} ranks")
        print(f"Rank {rank} using GPU {torch.cuda.current_device()}")

    hidden_size = 7168
    dtype = torch.bfloat16
    tensor_parallel_size = world_size
    eps = 1e-5

    torch.manual_seed(42)

    # Track if this rank failed
    rank_failed = False
    failure_message = ""

    try:
        if rank == 0:
            print(
                f"Testing seq_len={seq_len}, hidden_size={hidden_size}, fusion={fusion}"
            )

        # Generate test data (same on all ranks due to same seed)
        x_full = torch.randn(
            (tensor_parallel_size, seq_len, hidden_size),
            dtype=dtype,
            device=torch.device("cuda"),
        )
        residual = torch.randn(
            (seq_len, hidden_size), dtype=dtype, device=torch.device("cuda")
        )
        norm_weight = torch.randn(
            (hidden_size,), dtype=dtype, device=torch.device("cuda")
        )

        # Each rank gets its slice of the input
        x = x_full[rank, :, :]

        # Compute reference output based on fusion mode
        if fusion:
            # Fused case: AllReduce + Residual Add + RMS Norm
            allreduce_result = torch.sum(x_full, dim=0)  # AllReduce result
            residual_out = allreduce_result + residual  # Add residual
            print(
                "Device of residual_out:{}, norm_weight:{}".format(
                    residual_out.device, norm_weight.device
                )
            )
            norm_out = rmsnorm(residual_out, norm_weight, eps, enable_pdl=False)

            reference_output = (norm_out, residual_out)
        else:
            # Non-fused case: Only AllReduce
            allreduce_result = torch.sum(x_full, dim=0)  # AllReduce result
            reference_output = (allreduce_result,)

        # Run the test
        row_linear_residual_norm_fusion_forward(
            x,
            residual,
            norm_weight,
            eps,
            hidden_size,
            dtype,
            tensor_parallel_size,
            rank,
            fusion,
            reference_output,
        )

        # Synchronize before next test
        trtllm_mnnvl_ar.mpi_barrier()

        print(f"PASSED[rank={rank}]: seq_len={seq_len}, fusion={fusion}")

    except Exception as e:
        rank_failed = True
        failure_message = (
            f"FAILED[rank={rank}]: seq_len={seq_len}, fusion={fusion} failed: {e}"
        )
        print(failure_message)
        # Gather failure status from all ranks
        all_failures = MPI.COMM_WORLD.allgather(rank_failed)

        # If any rank failed, fail the test
        if any(all_failures):
            failed_ranks = [i for i, failed in enumerate(all_failures) if failed]
            if rank == 0:
                print(f"Test failed on ranks: {failed_ranks}")

            # Fail the test on all ranks
            pytest.fail(f"Test failed on ranks {failed_ranks}")
            trtllm_mnnvl_ar.mpi_barrier()

    # Final synchronization and check for failures across all ranks
    trtllm_mnnvl_ar.mpi_barrier()
