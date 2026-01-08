# Check torch version:
import traceback
from typing import Tuple, Optional

import pytest
import torch
import torch.distributed as dist

import flashinfer.comm.trtllm_mnnvl_ar as trtllm_mnnvl_ar
from flashinfer.comm.mapping import Mapping
from flashinfer.comm.mnnvl import TorchDistBackend

# Use flashinfer.norm.rmsnorm as reference implementation.
from flashinfer.norm import rmsnorm

# Test helpers
from tests.test_helpers.comm import (
    init_torch_distributed_from_mpi,
)

# Note: torch.distributed cleanup is handled by tests/comm/conftest.py


@torch.inference_mode()
def row_linear_residual_norm_fusion_forward(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    mapping: Mapping,
    fusion: bool,
    reference_output: tuple[torch.Tensor, ...],
    workspace: trtllm_mnnvl_ar.MNNVLAllReduceFusionWorkspace,
):
    tensor_parallel_rank = mapping.tp_rank
    dist.barrier()

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
            dist.barrier()

            output, residual_out = (
                trtllm_mnnvl_ar.trtllm_mnnvl_fused_allreduce_add_rmsnorm(
                    input,
                    residual,
                    norm_weight,
                    workspace,
                    eps,
                    launch_with_pdl=use_pdl,
                    strategy=trtllm_mnnvl_ar.MNNVLAllreduceFusionStrategy.AUTO,
                )
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


@torch.inference_mode()
def row_linear_residual_norm_fusion_forward_legacy(
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
    tensor_parallel_size = mapping.tp_size
    tensor_parallel_rank = mapping.tp_rank
    dist.barrier()

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
        input = input.view(-1, shape[-1])
        buffer_M = max_num_elements_mnnvl // hidden_size

        if enable_fusion:
            use_pdl = True

            prenorm_output = torch.empty_like(residual)
            normed_output = torch.empty_like(residual)

            dist.barrier()

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


def prepare_test_data(seq_len: int, hidden_size: int, dtype: torch.dtype, fusion: bool):
    # Use torch.distributed for communication between ranks
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        x_full = torch.randn((world_size, seq_len, hidden_size), dtype=dtype)
        residual = torch.randn((seq_len, hidden_size), dtype=dtype)
        norm_weight = torch.randn((hidden_size,), dtype=dtype)
    else:
        x_full = None
        residual = None
        norm_weight = None

    # Use torch.distributed broadcast_object_list for Python object broadcasting
    data_list = [x_full, residual, norm_weight]
    dist.broadcast_object_list(data_list, src=0)
    x_full, residual, norm_weight = data_list

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
    monkeypatch,
    seq_lens: list[int],
    fusion: bool,
    dtype: torch.dtype,
    hidden_size: int,
    legacy_explicit_workspace_bytes: Optional[int] = None,
    legacy_api: bool = False,
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

    gpus_per_node = torch.cuda.device_count()

    if gpus_per_node == 0:
        pytest.skip("MNNVL allreduce test requires at least one CUDA device per node")

    # Initialize torch.distributed (safe to call if already initialized)
    init_torch_distributed_from_mpi()

    # Get rank info from torch.distributed
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size < 2:
        pytest.skip(f"This test requires at least 2 ranks, got {world_size}")

    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        gpus_per_node=gpus_per_node,
        tp_size=world_size,
    )

    # Set CUDA device based on rank
    torch.cuda.set_device(mapping.local_rank)

    # Create TorchDistBackend for workspace creation (non-MPI based)
    comm_backend = TorchDistBackend()

    if mapping.local_rank == 0:
        print(
            f"[Node {mapping.node_rank}] Running MNNVL AllReduce test with {world_size} ranks"
        )
        print(
            f"[Node {mapping.node_rank}] Rank {rank} using GPU {torch.cuda.current_device()}"
        )
    eps = 1e-5
    torch.manual_seed(42 + rank)

    # Track if this rank failed
    rank_failed = False
    failure_message = ""

    try:
        if legacy_api:
            mcast_buffer_mnnvl, buffer_flags_mnnvl, max_num_elements_mnnvl = (
                trtllm_mnnvl_ar.get_allreduce_mnnvl_workspace(
                    mapping,
                    dtype,
                    comm_backend_for_handle_transfer=comm_backend,
                    buffer_size_in_bytes=legacy_explicit_workspace_bytes,
                )
            )

            multicast_ptr = mcast_buffer_mnnvl.get_multicast_ptr()
            buffer_ptrs_dev = mcast_buffer_mnnvl.get_buffer_ptrs_dev()
            unicast_ptr = mcast_buffer_mnnvl.mcast_device_memory.get_unicast_ptr(
                mapping.tp_rank
            )

        else:
            workspace = trtllm_mnnvl_ar.MNNVLAllReduceFusionWorkspace(
                mapping,
                max_num_tokens=max(seq_lens),
                hidden_dim=hidden_size,
                dtype=dtype,
                comm_backend=comm_backend,
            )

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
                print(
                    f"Testing seq_len={seq_len}, hidden_size={hidden_size}, fusion={fusion}, dtype={dtype}"
                )
            if legacy_api:
                row_linear_residual_norm_fusion_forward_legacy(
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
            else:
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

            # Synchronize before next test using torch.distributed barrier
            dist.barrier()

            print(
                f"PASSED[rank={rank}]: seq_len={seq_len}, fusion={fusion}, dtype={dtype}"
            )

    except Exception as e:
        rank_failed = True
        failure_message = f"FAILED[rank={rank}]: seq_lens={seq_lens}, fusion={fusion}, dtype={dtype} failed: {e}"
        print(failure_message)
        print(traceback.format_exc())

        # Gather failure status from all ranks using torch.distributed
        all_failures = [None] * world_size
        dist.all_gather_object(all_failures, rank_failed)

        if any(all_failures):
            failed_ranks = [i for i, failed in enumerate(all_failures) if failed]
            if rank == 0:
                print(f"Test failed on ranks: {failed_ranks}")

        # Re-raise the original exception so it can be caught by pytest.raises in negative tests
        raise

    finally:
        # Explicitly destroy workspace to avoid __del__ issues during Python shutdown
        if "workspace" in locals() and workspace is not None:
            workspace.destroy()
        if "mcast_buffer_mnnvl" in locals():
            del mcast_buffer_mnnvl

    # Final synchronization using torch.distributed barrier
    dist.barrier()


"""Test with default workspace size"""

# Multi-gpu test: mpirun -np 4 pytest tests/comm/test_trtllm_mnnvl_allreduce.py -vv -s
# Multi-node test:srun -A coreai_libraries_cudnn -N4 --container-image=<flashinfer_image> -J --mpi=pmix -- bash -c 'hostname && cd <path_to_flashinfer> && pip install -e . && python -m pytest tests/comm/test_trtllm_mnnvl_allreduce.py'


@pytest.mark.parametrize(
    "seq_lens",
    [[1], [4], [15], [27, 11, 24, 256], [127], [998, 2048]],
)
@pytest.mark.parametrize("fusion", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [2880, 5120, 7168, 8192, 16384])
def test_mnnvl_allreduce_refactored(
    monkeypatch, seq_lens: list[int], fusion: bool, dtype: torch.dtype, hidden_size: int
):
    """Test MNNVL AllReduce with refactored API."""
    run_mnnvl_ar_full(
        monkeypatch, seq_lens, fusion, dtype, hidden_size, legacy_api=False
    )


@pytest.mark.parametrize("seq_lens", [[1], [4], [15], [27, 11, 24], [127]])
@pytest.mark.parametrize("fusion", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [2048, 4096, 5120, 7168, 8192, 16384])
def test_mnnvl_allreduce_legacy(
    monkeypatch, seq_lens: list[int], fusion: bool, dtype: torch.dtype, hidden_size: int
):
    """Test MNNVL AllReduce with legacy API."""
    run_mnnvl_ar_full(
        monkeypatch, seq_lens, fusion, dtype, hidden_size, legacy_api=True
    )
