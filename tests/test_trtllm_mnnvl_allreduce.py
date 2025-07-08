# Check torch version:
import os
import sys
import traceback

import torch
from mpi4py import MPI  # Added MPI import

import flashinfer.comm as comm
from flashinfer.comm.mapping import Mapping
from flashinfer.comm.mnnvl import McastDeviceMemory, McastGPUBuffer


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
    os.environ["TRTLLM_MNNVL_AR_ENABLED"] = "1"

    mapping = Mapping(
        world_size=tensor_parallel_size,
        tp_size=tensor_parallel_size,
        rank=tensor_parallel_rank,
    )

    def func(input, residual, norm_weight, eps, enable_fusion, buffer_mnnvl):
        # For both fused and unfused cases:

        multicast_ptr = mcast_buffer_mnnvl.get_multicast_ptr_as_int64()
        buffer_ptrs_dev = mcast_buffer_mnnvl.get_buffer_ptrs_dev_as_ctypes_ptr()

        shape = input.shape

        assert buffer_mnnvl.shape[-1] % shape[-1] == 0

        input = input.view(-1, shape[-1])
        output = torch.empty_like(input)
        buffer_mnnvl = buffer_mnnvl.view(3, 2, -1, shape[-1])

        if enable_fusion:
            raise NotImplementedError("Fusion not implemented")

        else:
            comm.trtllm_mnnvl_all_reduce(
                input,
                output,
                multicast_ptr,
                buffer_ptrs_dev,  # Attempted to use this raw pointer
                buffer_mnnvl,
                buffer_flags_mnnvl,
                tensor_parallel_size,
                tensor_parallel_rank,
                True,  # wait_for_results
                False,  # launch_with_pdl
            )
            return (output.view(shape),)

    # Get workspace buffers using MPI rank
    mcast_buffer_mnnvl, buffer_mnnvl, buffer_flags_mnnvl, max_num_elements_mnnvl = (
        comm.get_allreduce_mnnvl_workspace(mapping, dtype)
    )

    try:
        output = func(
            x.clone(), residual.clone(), norm_weight, eps, fusion, buffer_mnnvl
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


def test_mnnvl_allreduce_full(seq_len: int, fusion: bool):
    """Main test function that runs on each MPI rank"""

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

    try:
        if rank == 0:
            print(
                f"Testing seq_len={seq_len}, hidden_size={hidden_size}, fusion={fusion}"
            )

        # Generate test data (same on all ranks due to same seed)
        x_full = torch.randn((tensor_parallel_size, seq_len, hidden_size), dtype=dtype)
        residual = torch.randn((seq_len, hidden_size), dtype=dtype)
        norm_weight = torch.randn((hidden_size,), dtype=dtype)

        # Each rank gets its slice of the input
        x = x_full[rank, :, :]

        # Compute reference output based on fusion mode
        if fusion:
            raise NotImplementedError("Fusion not implemented")
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
        comm.mpi_barrier()

        print(f"PASSED[rank={rank}]: seq_len={seq_len}, fusion={fusion}")

    except Exception as e:
        print(f"FAILED[rank={rank}]: seq_len={seq_len}, fusion={fusion} failed: {e}")
        if rank == 0:
            traceback.print_exc()
        # Don't exit immediately, let other tests run
        comm.mpi_barrier()

    # Final synchronization and results
    comm.mpi_barrier()


if __name__ == "__main__":
    os.environ["TRTLLM_FORCE_MNNVL_AR"] = "1"  # force multi-node allreduce.
    # Test parameters
    # seq_lens = [1, 4, 32, 128]
    fusion_modes = [False]  # Only non-fused case for now
    for fusion in fusion_modes:
        test_mnnvl_allreduce_full(4, fusion)
