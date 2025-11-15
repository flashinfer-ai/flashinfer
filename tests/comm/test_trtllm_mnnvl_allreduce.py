# Check torch version:
from typing import Tuple

import pytest
import torch
from mpi4py import MPI  # Added MPI import

import flashinfer.comm.trtllm_mnnvl_ar as trtllm_mnnvl_ar
from flashinfer.comm.mapping import Mapping

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
    mapping: Mapping,
    fusion: bool,
    reference_output: tuple[torch.Tensor, ...],
    multicast_ptr: int,
    buffer_ptrs_dev: int,
    unicast_ptr: int,
    max_num_elements_mnnvl: int,
    buffer_flags_mnnvl: torch.Tensor,
):
    x = x.cuda()
    residual = residual.cuda()
    norm_weight = norm_weight.cuda()
    reference_output = tuple(t.cuda() for t in reference_output)

    tensor_parallel_size = mapping.tp_size
    tensor_parallel_rank = mapping.tp_rank

    MPI.COMM_WORLD.barrier()

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


"""Helper function to run the core MNNVL AllReduce test logic"""


def run_mnnvl_ar_full(
    monkeypatch,
    seq_lens: list[int],
    fusion: bool,
    dtype: torch.dtype,
    hidden_size: int,
    explicit_workspace_bytes: int | None = None,
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
    monkeypatch.setenv("TRTLLM_FORCE_MNNVL_AR", "1")  # force multi-node allreduce.

    # Get MPI info
    rank = MPI.COMM_WORLD.Get_rank()
    world_size = MPI.COMM_WORLD.Get_size()
    gpus_per_node = torch.cuda.device_count()

    if gpus_per_node == 0:
        pytest.skip("MNNVL allreduce test requires at least one CUDA device per node")

    # Ensure we have exactly 2 ranks for this test
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
        print(
            f"[Node {mapping.node_rank}] Running MNNVL AllReduce test with {world_size} ranks"
        )
        print(
            f"[Node {mapping.node_rank}] Rank {rank} using GPU {torch.cuda.current_device()}"
        )

    tensor_parallel_size = world_size
    eps = 1e-5
    torch.manual_seed(42)

    # Track if this rank failed
    rank_failed = False
    failure_message = ""

    try:
        # Get workspace buffers using MPI rank - allocate once per seq_lens list and reuse within the list
        # This workspace is sized for the maximum expected sequence length and can be reused within each list
        # Each parameterized list gets its own fresh workspace allocation
        mcast_buffer_mnnvl, buffer_flags_mnnvl, max_num_elements_mnnvl = (
            trtllm_mnnvl_ar.get_allreduce_mnnvl_workspace(
                mapping, dtype, buffer_size_in_bytes=explicit_workspace_bytes
            )
        )

        multicast_ptr = mcast_buffer_mnnvl.get_multicast_ptr()
        buffer_ptrs_dev = mcast_buffer_mnnvl.get_buffer_ptrs_dev()
        unicast_ptr = mcast_buffer_mnnvl.mcast_device_memory.get_unicast_ptr(
            mapping.tp_rank
        )

        # Test each sequence length with the same workspace (reusing allocated buffers within this list)
        for seq_len in seq_lens:
            if rank == 0:
                print(
                    f"Testing seq_len={seq_len}, hidden_size={hidden_size}, fusion={fusion}, dtype={dtype}"
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
            reference_output: Tuple[torch.Tensor, ...] = None
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

            # Run the test with the same workspace
            row_linear_residual_norm_fusion_forward(
                x,
                residual,
                norm_weight,
                eps,
                hidden_size,
                dtype,
                mapping,
                fusion,
                reference_output,
                multicast_ptr,
                buffer_ptrs_dev,
                unicast_ptr,
                max_num_elements_mnnvl,
                buffer_flags_mnnvl,
            )

            # Synchronize before next test
            trtllm_mnnvl_ar.mpi_barrier()

            print(
                f"PASSED[rank={rank}]: seq_len={seq_len}, fusion={fusion}, dtype={dtype}"
            )

    except Exception as e:
        rank_failed = True
        failure_message = f"FAILED[rank={rank}]: seq_lens={seq_lens}, fusion={fusion}, dtype={dtype} failed: {e}"
        print(failure_message)

        # Gather failure status from all ranks for logging
        all_failures = MPI.COMM_WORLD.allgather(rank_failed)

        if any(all_failures):
            failed_ranks = [i for i, failed in enumerate(all_failures) if failed]
            if rank == 0:
                print(f"Test failed on ranks: {failed_ranks}")

        # Cleanup before re-raising
        if "mcast_buffer_mnnvl" in locals():
            del mcast_buffer_mnnvl

        # Re-raise the original exception so it can be caught by pytest.raises in negative tests
        raise

    finally:
        # Ensure cleanup happens for this list's workspace
        if "mcast_buffer_mnnvl" in locals():
            del mcast_buffer_mnnvl

    # Final synchronization and check for failures across all ranks
    trtllm_mnnvl_ar.mpi_barrier()


"""Test with default workspace size"""


@pytest.mark.parametrize(
    "seq_lens",
    [
        [1],
        [4],
        [15],
        [27, 11, 24],
        [127],
    ],
)
@pytest.mark.parametrize("fusion", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [2048, 4096, 5120, 7168, 8192])
def test_mnnvl_allreduce_default_workspace(
    monkeypatch, seq_lens: list[int], fusion: bool, dtype: torch.dtype, hidden_size: int
):
    """Test MNNVL AllReduce with default workspace size."""
    run_mnnvl_ar_full(monkeypatch, seq_lens, fusion, dtype, hidden_size)


"""Test with explicit workspace size"""


@pytest.mark.parametrize(
    "seq_lens",
    [
        [1, 4, 180],
    ],
)
@pytest.mark.parametrize("fusion", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [2048, 4096, 5120, 7168, 8192])
def test_mnnvl_allreduce_explicit_workspace(
    monkeypatch, seq_lens: list[int], fusion: bool, dtype: torch.dtype, hidden_size: int
):
    """Test MNNVL AllReduce with explicitly calculated workspace size."""
    # Calculate workspace to fit the maximum sequence length
    # buffer shape: [3, 2, buffer_tokens, hidden_dim]
    explicit_workspace_bytes = 3 * 2 * dtype.itemsize * hidden_size * max(seq_lens)
    run_mnnvl_ar_full(
        monkeypatch,
        seq_lens,
        fusion,
        dtype,
        hidden_size,
        explicit_workspace_bytes=explicit_workspace_bytes,
    )


"""Negative test: workspace too small"""


@pytest.mark.parametrize("fusion", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [2048, 4096])
def test_mnnvl_allreduce_workspace_too_small(
    monkeypatch, fusion: bool, dtype: torch.dtype, hidden_size: int
):
    """Test that MNNVL AllReduce fails gracefully when workspace is too small."""
    # Use a large sequence length that won't fit in a small workspace
    seq_len = 180

    # Create a workspace that's too small (only enough for 10 tokens)
    small_workspace_bytes = 3 * 2 * dtype.itemsize * hidden_size * 10

    # Expect a ValueError with a message about buffer_M being too small
    with pytest.raises((ValueError, RuntimeError)) as exc_info:
        run_mnnvl_ar_full(
            monkeypatch,
            [seq_len],
            fusion,
            dtype,
            hidden_size,
            explicit_workspace_bytes=small_workspace_bytes,
        )

    # Verify the error message contains the expected text
    assert "greater than the buffer_M" in str(exc_info.value)
