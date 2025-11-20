# Check torch version:
from typing import Tuple, Optional

import pytest
import torch
from mpi4py import MPI  # Added MPI import

import flashinfer.comm.trtllm_mnnvl_ar as trtllm_mnnvl_ar
from flashinfer.comm.mapping import Mapping
from flashinfer.comm.mnnvl import CommBackend, MpiComm

# Use flashinfer.norm.rmsnorm as reference implementation.
from flashinfer.norm import rmsnorm


@torch.inference_mode()
def row_linear_residual_norm_fusion_forward(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    mapping: Mapping,
    fusion: bool,
    reference_output: tuple[torch.Tensor, ...],
    workspace: trtllm_mnnvl_ar.MNNVLAllreduceFusionWorkspace,
):
    tensor_parallel_rank = mapping.tp_rank
<<<<<<< HEAD
    if comm_backend_for_handle_transfer is None:
        comm = MpiComm()
    else:
        comm = comm_backend_for_handle_transfer
    comm.barrier()
=======
    MPI.COMM_WORLD.barrier()
>>>>>>> bca4f5d9 (Passing the test.)

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

            output, residual_out = trtllm_mnnvl_ar.trtllm_mnnvl_fused_allreduce_rmsnorm(
                input,
                residual,
                norm_weight,
                workspace,
                eps,
                launch_with_pdl=use_pdl,
                strategy=trtllm_mnnvl_ar.MNNVLAllreduceFusionStrategy.AUTO,
            )

            return output.view(shape), residual_out.view(shape)

        else:
            output = torch.empty_like(input)

            output = trtllm_mnnvl_ar.trtllm_mnnvl_allreduce(
                input,
                workspace,
                launch_with_pdl=use_pdl,
                strategy=trtllm_mnnvl_ar.MNNVLAllreduceFusionStrategy.AUTO,
            )
            return (output.view(shape),)

    output = func(x.clone(), residual.clone(), norm_weight, eps, fusion, workspace)

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


"""Helper function to run the core MNNVL AllReduce test logic"""


def prepare_test_data(seq_len: int, hidden_size: int, dtype: torch.dtype, fusion: bool):
    # Communicator used for passing data between ranks
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


def run_mnnvl_ar_full(
    monkeypatch, seq_lens: list[int], fusion: bool, dtype: torch.dtype, hidden_size: int
):
    """Core test logic for MNNVL AllReduce operations.

    Args:
        monkeypatch: pytest monkeypatch fixture
        seq_lens: List of sequence lengths to test
        fusion: Whether to test fused allreduce+rmsnorm or just allreduce
        dtype: Data type for tensors
        hidden_size: Hidden dimension size
        explicit_workspace_bytes: If provided, use this workspace size instead of default
    """

    comm = MPI.COMM_WORLD
    # Get MPI info
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    gpus_per_node = torch.cuda.device_count()

    if gpus_per_node == 0:
        pytest.skip("MNNVL allreduce test requires at least one CUDA device per node")
    if world_size < 2:
        pytest.skip(f"This test requires at least 2 MPI ranks, got {world_size}")

    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        gpus_per_node=gpus_per_node,
        tp_size=world_size,
    )

    # Set CUDA device based on rank
    torch.cuda.set_device(mapping.local_rank)

    if mapping.local_rank == 0:
<<<<<<< HEAD
        print(f"[Node {mapping.node_rank}] Running MNNVL AllReduce test with {world_size} ranks")
        print(f"[Node {mapping.node_rank}] Rank {rank} using GPU {torch.cuda.current_device()}")

    tensor_parallel_size = world_size
=======
        print(
            f"[Node {mapping.node_rank}] Running MNNVL AllReduce test with {world_size} ranks"
        )
        print(
            f"[Node {mapping.node_rank}] Rank {rank} using GPU {torch.cuda.current_device()}"
        )
>>>>>>> bca4f5d9 (Passing the test.)
    eps = 1e-5
    torch.manual_seed(42 + rank)

    # Track if this rank failed
    rank_failed = False
    failure_message = ""

    try:
        required_workspace_bytes = trtllm_mnnvl_ar.MNNVLAllreduceFusionWorkspace.get_required_buffer_size_bytes(
            mapping.tp_size,
            max(seq_lens),
            hidden_size,
            dtype,
            trtllm_mnnvl_ar.MNNVLAllreduceFusionStrategy.AUTO,
        )
        workspace = trtllm_mnnvl_ar.MNNVLAllreduceFusionWorkspace(mapping, required_workspace_bytes)

        test_data = []
        for seq_len in seq_lens:
            (x_local, residual, norm_weight), reference_output = prepare_test_data(
                seq_len, hidden_size, dtype, fusion
            )
            test_data.append(
                (seq_len, x_local, residual, norm_weight, reference_output)
            )

        # Test each sequence length with the same workspace (reusing allocated buffers within this list)
        for seq_len, x, residual, norm_weight, reference_output in test_data:
            if rank == 0:
<<<<<<< HEAD
                print(f"Testing seq_len={seq_len}, hidden_size={hidden_size}, fusion={fusion}, dtype={dtype}")
                print(f"[Rank {rank}] Buffer flags: {workspace.buffer_flags}")

            # Generate test data (same on all ranks due to same seed)
            x_full = torch.randn(
                (tensor_parallel_size, seq_len, hidden_size),
                dtype=dtype,
                device=torch.device("cuda"),
            )
            residual = torch.randn((seq_len, hidden_size), dtype=dtype, device=torch.device("cuda"))
            norm_weight = torch.randn((hidden_size,), dtype=dtype, device=torch.device("cuda"))

            # Each rank gets its slice of the input
            x = x_full[rank, :, :]

            # Compute reference output based on fusion mode
            reference_output: Tuple[torch.Tensor, ...] = None
            if fusion:
                # Fused case: AllReduce + Residual Add + RMS Norm
                allreduce_result = torch.sum(x_full, dim=0)  # AllReduce result
                residual_out = allreduce_result + residual  # Add residual
                print("Device of residual_out:{}, norm_weight:{}".format(residual_out.device, norm_weight.device))
                norm_out = rmsnorm(residual_out, norm_weight, eps, enable_pdl=False)

                reference_output = (norm_out, residual_out)
            else:
                # Non-fused case: Only AllReduce
                allreduce_result = torch.sum(x_full, dim=0)  # AllReduce result
                reference_output = (allreduce_result,)
=======
                print(
                    f"Testing seq_len={seq_len}, hidden_size={hidden_size}, fusion={fusion}, dtype={dtype}"
                )
>>>>>>> bca4f5d9 (Passing the test.)

            # Run the test with the same workspace
            row_linear_residual_norm_fusion_forward(
                x,
                residual,
                norm_weight,
                eps,
                mapping,
                fusion,
                reference_output,
                workspace,
            )

            # Synchronize before next test
            trtllm_mnnvl_ar.mpi_barrier()

            print(f"PASSED[rank={rank}]: seq_len={seq_len}, fusion={fusion}, dtype={dtype}")

    except Exception as e:
        rank_failed = True
        failure_message = f"FAILED[rank={rank}]: seq_lens={seq_lens}, fusion={fusion}, dtype={dtype} failed: {e}"
        print(failure_message)
        import traceback

        print(traceback.format_exc())

        # Gather failure status from all ranks for logging
        all_failures = MPI.COMM_WORLD.allgather(rank_failed)

        if any(all_failures):
            failed_ranks = [i for i, failed in enumerate(all_failures) if failed]
            if rank == 0:
                print(f"Test failed on ranks: {failed_ranks}")

        # Cleanup before re-raising
        if "workspace" in locals():
            del workspace

        # Re-raise the original exception so it can be caught by pytest.raises in negative tests
        raise

    finally:
        # Ensure cleanup happens for this list's workspace
        if "workspace" in locals():
            del workspace

    # Final synchronization and check for failures across all ranks
    trtllm_mnnvl_ar.mpi_barrier()


"""Test with default workspace size"""


@pytest.mark.parametrize(
    "seq_lens",
    [[1], [4], [15], [27, 11, 24, 256], [127], [998, 2048]],
)
@pytest.mark.parametrize("fusion", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [2880, 5120, 7168, 8192])
def test_mnnvl_allreduce_default_workspace(
    monkeypatch, seq_lens: list[int], fusion: bool, dtype: torch.dtype, hidden_size: int
):
    """Test MNNVL AllReduce with default workspace size."""
    run_mnnvl_ar_full(monkeypatch, seq_lens, fusion, dtype, hidden_size)
