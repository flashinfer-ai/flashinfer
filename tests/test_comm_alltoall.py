"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pytest
import torch

import flashinfer.comm as comm

has_setup_max_sm_count = False


@pytest.fixture(autouse=True, scope="session")
def setup_test_environment():
    """Set up test environment and warm up JIT compilation."""
    global has_setup_max_sm_count
    if not has_setup_max_sm_count:
        # Set up SM count once for all tests
        sm_count = torch.cuda.get_device_properties(0).multi_processor_count
        max_sm_count = sm_count // 8  # Maximum world size is 8
        comm.set_moe_max_usable_sm_count(max_sm_count)
        has_setup_max_sm_count = True

    torch.manual_seed(0x1234)
    yield


# Single GPU test parameters
SINGLE_GPU_PARAMS = [
    (902, 701, 32768, 100, torch.float16),  # Large data, float16
    (101, 75, 288, 10, torch.float16),  # Medium data, float16
    (10, 5, 8, 1, torch.float16),  # Small data, float16
    (902, 701, 7168, 100, torch.bfloat16),  # Large data, bfloat16
    (101, 75, 288, 10, torch.bfloat16),  # Medium data, bfloat16
]

MULTI_RANK_PARAMS = [
    (2, 5, 8, torch.float16),  # Small input, 2 ranks
    (4, 901, 32768, torch.bfloat16),  # Large input, 4 ranks
    (8, 16384, 128, torch.float16),  # Many small vectors, 8 ranks
]

PREPARE_INDICES_PARAMS = [
    (0, 8, 256, 4, 3, False),  # Rank 0, small config
    (1, 8, 256, 4, 3, True),  # Rank 1, small config with real cumsum
    (7, 8, 256, 8, 1025, False),  # High rank, medium config
    (7, 64, 1024, 32, 1029, True),  # High rank, large config with real cumsum
]

LOCAL_GATHER_PARAMS = [
    (0, 8, 256, 4, 3),  # Rank 0, small config
    (7, 8, 256, 8, 32),  # High rank, medium config
    (7, 64, 1024, 32, 1029),  # High rank, large config
]


# Real cross-GPU communication test parameters
CROSS_GPU_PARAMS = [
    (2, 100, 256, torch.float16),  # 2 GPUs, 2 ranks
    (2, 300, 512, torch.bfloat16),  # 2 GPUs, 2 ranks, larger data
    (4, 150, 256, torch.float16),  # 4 GPUs, 4 ranks (if available)
    (4, 400, 512, torch.float16),  # 4 GPUs, 4 ranks, larger data
]


def get_available_gpu_count():
    """Get the number of available GPUs."""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def requires_gpus(min_gpus):
    """Decorator to skip test if insufficient GPUs are available."""

    def decorator(func):
        return pytest.mark.skipif(
            get_available_gpu_count() < min_gpus,
            reason=f"Requires at least {min_gpus} GPUs, but only {get_available_gpu_count()} available",
        )(func)

    return decorator


@pytest.mark.parametrize(
    "input_entry_count,output_entry_count,vector_dim,send_recv_count,dtype",
    SINGLE_GPU_PARAMS,
)
def test_moe_alltoall_single_gpu(
    input_entry_count, output_entry_count, vector_dim, send_recv_count, dtype
):
    """Test MOE alltoall communication on single GPU."""
    torch.cuda.set_device(0)
    # Create a random input tensor
    input_tensor = torch.randn(
        input_entry_count, vector_dim, dtype=dtype, device=torch.device("cuda")
    )
    output_tensor = torch.zeros(
        output_entry_count, vector_dim, dtype=dtype, device=torch.device("cuda")
    )

    send_cumsum = (
        torch.ones((1,), dtype=torch.int32, device=torch.device("cuda"))
        * send_recv_count
    )
    recv_cumsum = (
        torch.ones((1,), dtype=torch.int32, device=torch.device("cuda"))
        * send_recv_count
    )
    send_indices = torch.randperm(
        input_entry_count, dtype=torch.int32, device=torch.device("cuda")
    )[:send_recv_count]
    recv_indices = torch.randperm(
        output_entry_count, dtype=torch.int32, device=torch.device("cuda")
    )[:send_recv_count]

    ref_output_tensor = torch.zeros(
        output_entry_count, vector_dim, dtype=dtype, device=torch.device("cuda")
    )
    ref_output_tensor[recv_indices] = input_tensor[send_indices]

    workspace_size = comm.get_moe_commworkspace_size_per_rank(1)
    all_workspaces = torch.zeros(
        1, workspace_size, dtype=torch.uint64, device=torch.device("cuda")
    )

    comm.moe_comm(
        input_tensor,
        send_cumsum,
        send_indices,
        output_tensor,
        recv_cumsum,
        recv_indices,
        all_workspaces,
        0,
        1,
    )

    torch.testing.assert_close(output_tensor, ref_output_tensor, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "world_size,input_entry_per_rank,vector_dim,dtype", MULTI_RANK_PARAMS
)
def test_moe_alltoall_multi_rank_single_gpu(
    world_size, input_entry_per_rank, vector_dim, dtype
):
    """Test MOE alltoall communication with multiple ranks on single GPU."""
    torch.cuda.set_device(0)
    max_world_size = 8
    assert (
        world_size <= max_world_size
    ), f"should run with world_size at most {max_world_size}"

    # SM count is now set up globally in the fixture

    # Create a random input tensor
    input_tensor = torch.randn(
        input_entry_per_rank * world_size,
        vector_dim,
        dtype=dtype,
        device=torch.device("cuda"),
    )
    output_tensor = torch.zeros(
        input_entry_per_rank * world_size,
        vector_dim,
        dtype=dtype,
        device=torch.device("cuda"),
    )
    ref_output_tensor = torch.zeros(
        input_entry_per_rank * world_size,
        vector_dim,
        dtype=dtype,
        device=torch.device("cuda"),
    )
    target_rank_ids = torch.randint(
        0,
        world_size,
        (input_entry_per_rank * world_size,),
        dtype=torch.int32,
        device=torch.device("cuda"),
    )

    input_tensors_all_ranks = list(torch.split(input_tensor, input_entry_per_rank))
    target_rank_ids_all_ranks = list(torch.split(target_rank_ids, input_entry_per_rank))

    send_ids_all_ranks = []
    send_counts_all_ranks = []
    send_cumsum_all_ranks = []
    send_start_end_all_ranks = []

    # each rank do its own local compute to get how to send data to other ranks.
    for rank in range(world_size):
        send_start_end = []
        local_target_rank_ids = target_rank_ids_all_ranks[rank]
        sorted_local_target_rank_ids, local_send_id = torch.sort(local_target_rank_ids)
        local_send_id = local_send_id.to(torch.int32)
        padded_sorted_local_target_rank_ids = torch.cat(
            (
                sorted_local_target_rank_ids,
                torch.arange(
                    world_size, dtype=torch.int32, device=torch.device("cuda")
                ),
            )
        )
        unique_target_rank_ids, local_send_counts = torch.unique(
            padded_sorted_local_target_rank_ids, return_counts=True
        )
        local_send_counts = local_send_counts.to(torch.int32)
        assert (
            unique_target_rank_ids.numel() == world_size
        ), "unique_target_rank_ids must be equal to world_size"
        local_send_counts -= 1  # remove padding
        local_send_cumsum = torch.cumsum(local_send_counts, dim=0).to(torch.int32)
        send_ids_all_ranks.append(local_send_id)
        send_counts_all_ranks.append(local_send_counts)
        send_cumsum_all_ranks.append(local_send_cumsum)
        local_send_cumsum_cpu = local_send_cumsum.cpu().tolist()
        for i in range(len(local_send_cumsum_cpu)):
            send_start_end.append(
                (
                    local_send_cumsum_cpu[i - 1] if i > 0 else 0,
                    local_send_cumsum_cpu[i],
                )
            )
        send_start_end_all_ranks.append(send_start_end)

    recv_ids_all_ranks = []
    recv_cumsum_all_ranks = []

    output_tensors_all_ranks = []

    total_recv_all_ranks_cpu = []
    output_indice_offset = 0

    output_start_current_rank = 0
    # each rank do compute based on other ranks' send counts to get how to receive data from other ranks.
    for rank in range(world_size):
        local_recv_counts = torch.zeros(
            world_size, dtype=torch.int32, device=torch.device("cuda")
        )
        for other_rank in range(world_size):
            local_recv_counts[other_rank] = send_counts_all_ranks[other_rank][rank]
            local_recv_count_pair = local_recv_counts[other_rank].cpu().item()
            send_rank_start_end = send_start_end_all_ranks[other_rank][rank]
            ref_output_tensor[
                output_indice_offset : output_indice_offset + local_recv_count_pair
            ] = input_tensors_all_ranks[other_rank][
                send_ids_all_ranks[other_rank][
                    send_rank_start_end[0] : send_rank_start_end[1]
                ]
            ]
            output_indice_offset += local_recv_count_pair
        local_recv_cumsum = torch.cumsum(local_recv_counts, dim=0).to(torch.int32)
        recv_cumsum_all_ranks.append(local_recv_cumsum)
        total_recv_count = local_recv_cumsum[-1].cpu()
        total_recv_all_ranks_cpu.append(total_recv_count)
        output_tensors_all_ranks.append(
            output_tensor[
                output_start_current_rank : output_start_current_rank + total_recv_count
            ]
        )
        output_start_current_rank += total_recv_count
        local_recv_ids = torch.arange(
            total_recv_count, dtype=torch.int32, device=torch.device("cuda")
        )
        recv_ids_all_ranks.append(local_recv_ids)

    cuda_streams_all_ranks = [torch.cuda.Stream() for _ in range(world_size)]

    workspace_size = comm.get_moe_commworkspace_size_per_rank(world_size)
    all_workspaces = torch.zeros(
        world_size, workspace_size, dtype=torch.uint64, device=torch.device("cuda")
    )

    # Synchronize before starting parallel communication
    torch.cuda.synchronize()

    # do alltoall in parallel
    for rank in range(world_size):
        with torch.cuda.stream(cuda_streams_all_ranks[rank]):
            comm.moe_comm(
                input_tensors_all_ranks[rank],
                send_cumsum_all_ranks[rank],
                send_ids_all_ranks[rank],
                output_tensors_all_ranks[rank],
                recv_cumsum_all_ranks[rank],
                recv_ids_all_ranks[rank],
                all_workspaces,
                rank,
                world_size,
            )
    for rank in range(world_size):
        cuda_streams_all_ranks[rank].synchronize()

    torch.testing.assert_close(output_tensor, ref_output_tensor, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "ep_rank,ep_size,expert_count,top_k,max_token_count_per_rank,use_real_rank_token_count_cumsum",
    PREPARE_INDICES_PARAMS,
)
def test_moe_alltoall_prepare_indices(
    ep_rank,
    ep_size,
    expert_count,
    top_k,
    max_token_count_per_rank,
    use_real_rank_token_count_cumsum,
):
    """Test MOE alltoall prepare indices functionality."""
    torch.cuda.set_device(0)

    def generate_references():
        rank_token_count = max_token_count_per_rank
        if use_real_rank_token_count_cumsum:
            # Make sure we have at least 1 token in each rank except last rank
            rank_token_counts = [
                max(1, torch.randint(1, max_token_count_per_rank + 1, (1,)).item())
                for _ in range(ep_size - 1)
            ]
            rank_token_counts.append(
                max_token_count_per_rank
            )  # last rank has max tokens
            real_rank_token_count_cumsum = (
                torch.tensor(
                    rank_token_counts, dtype=torch.int32, device=torch.device("cuda")
                )
                .cumsum(dim=0)
                .to(torch.int32)
            )
            rank_token_count = rank_token_counts[ep_rank]
        else:
            real_rank_token_count_cumsum = None

        # Generate target rank ids for this rank
        target_rank_ids = torch.randint(
            0,
            ep_size,
            (rank_token_count, top_k),
            dtype=torch.int32,
            device=torch.device("cuda"),
        )

        if not use_real_rank_token_count_cumsum:
            gathered_target_rank_ids = torch.zeros(
                ep_size * max_token_count_per_rank,
                top_k,
                dtype=torch.int32,
                device=torch.device("cuda"),
            )
            gathered_target_rank_ids[
                ep_rank * max_token_count_per_rank : ep_rank * max_token_count_per_rank
                + rank_token_count
            ] = target_rank_ids
        else:
            total_tokens = real_rank_token_count_cumsum[-1].item()
            gathered_target_rank_ids = torch.zeros(
                total_tokens, top_k, dtype=torch.int32, device=torch.device("cuda")
            )
            start_pos = (
                0 if ep_rank == 0 else real_rank_token_count_cumsum[ep_rank - 1].item()
            )
            gathered_target_rank_ids[start_pos : start_pos + rank_token_count] = (
                target_rank_ids
            )

        return gathered_target_rank_ids, real_rank_token_count_cumsum, target_rank_ids

    gathered_target_rank_ids, real_rank_token_count_cumsum, target_rank_ids = (
        generate_references()
    )

    (
        local_gather_indices,
        send_rank_count_cumsum,
        send_rank_local_indices,
        recv_rank_count_cumsum,
        recv_rank_local_indices,
        backward_recv_rank_local_indices,
    ) = comm.moe_comm_prepare_indices(
        gathered_target_rank_ids,
        real_rank_token_count_cumsum,
        max_token_count_per_rank,
        expert_count,
        top_k,
        ep_rank,
        ep_size,
    )

    # Validate shapes
    assert local_gather_indices.shape[0] <= max_token_count_per_rank * ep_size
    assert send_rank_count_cumsum.shape[0] == ep_size
    assert recv_rank_count_cumsum.shape[0] == ep_size
    assert send_rank_local_indices.shape[0] <= max_token_count_per_rank * max(
        ep_size, top_k
    )
    assert recv_rank_local_indices.shape[0] <= max_token_count_per_rank * ep_size
    assert backward_recv_rank_local_indices.shape[0] <= max_token_count_per_rank * max(
        ep_size, top_k
    )

    # Basic validation - cumulative sums should be non-decreasing
    assert torch.all(send_rank_count_cumsum[1:] >= send_rank_count_cumsum[:-1])
    assert torch.all(recv_rank_count_cumsum[1:] >= recv_rank_count_cumsum[:-1])


@pytest.mark.parametrize(
    "ep_rank,ep_size,expert_count,top_k,max_token_count_per_rank", LOCAL_GATHER_PARAMS
)
def test_moe_local_gather(
    ep_rank,
    ep_size,
    expert_count,
    top_k,
    max_token_count_per_rank,
):
    """Test MOE local gather functionality."""
    torch.cuda.set_device(0)

    # Generate test data using the original method
    rank_token_count_cumsum = torch.randint(
        0,
        max_token_count_per_rank + 1,
        (ep_size,),
        dtype=torch.int32,
        device=torch.device("cuda"),
    )
    rank_token_count_cumsum = torch.cumsum(rank_token_count_cumsum, dim=0).to(
        torch.int32
    )
    local_token_count = rank_token_count_cumsum[ep_size - 1].cpu().item()
    local_max_token_count = max_token_count_per_rank * ep_size
    local_gather_indices = torch.randint(
        0,
        max_token_count_per_rank * ep_size,
        (local_max_token_count,),
        dtype=torch.int32,
        device=torch.device("cuda"),
    )

    gathered_expert_ids = torch.randint(
        0,
        expert_count,
        (max_token_count_per_rank * ep_size, top_k),
        dtype=torch.int32,
        device=torch.device("cuda"),
    )
    gathered_scales = torch.rand(
        (max_token_count_per_rank * ep_size, top_k),
        dtype=torch.float32,
        device=torch.device("cuda"),
    )

    ref_local_expert_ids = torch.zeros(
        local_max_token_count, top_k, dtype=torch.int32, device=torch.device("cuda")
    )
    ref_local_scales = torch.zeros(
        local_max_token_count,
        top_k,
        dtype=torch.float32,
        device=torch.device("cuda"),
    )

    # compute reference
    ref_local_expert_ids += expert_count
    valid_local_gather_indices = local_gather_indices[:local_token_count]
    ref_local_expert_ids[:local_token_count] = gathered_expert_ids[
        valid_local_gather_indices
    ]
    ref_local_scales[:local_token_count] = gathered_scales[valid_local_gather_indices]

    local_expert_ids = torch.empty(
        local_max_token_count, top_k, dtype=torch.int32, device=torch.device("cuda")
    )
    local_scales = torch.empty(
        local_max_token_count,
        top_k,
        dtype=torch.float32,
        device=torch.device("cuda"),
    )

    comm.moe_local_gather(
        rank_token_count_cumsum,
        local_gather_indices,
        gathered_expert_ids,
        gathered_scales,
        local_expert_ids,
        local_scales,
        max_token_count_per_rank,
        expert_count,
        top_k,
        ep_rank,
        ep_size,
    )

    assert torch.equal(local_expert_ids, ref_local_expert_ids)
    assert torch.equal(local_scales, ref_local_scales)


def verify_cross_gpu_data_movement(
    input_tensors_per_rank,
    output_tensors_per_rank,
    send_patterns,
    recv_patterns,
    world_size,
):
    """Verify that data actually moved between different GPUs."""
    total_data_moved = 0
    for sender_rank in range(world_size):
        for receiver_rank in range(world_size):
            if sender_rank == receiver_rank:
                continue  # Skip self-communication

            # Get the amount of data sent from sender to receiver
            send_counts = send_patterns[sender_rank]
            recv_counts = recv_patterns[receiver_rank]

            sender_to_receiver_count = send_counts[receiver_rank].cpu().item()
            receiver_from_sender_count = recv_counts[sender_rank].cpu().item()

            assert sender_to_receiver_count == receiver_from_sender_count, (
                f"Mismatch: rank {sender_rank} sends {sender_to_receiver_count} to rank {receiver_rank}, "
                f"but rank {receiver_rank} expects {receiver_from_sender_count} from rank {sender_rank}"
            )

            if sender_to_receiver_count > 0:
                total_data_moved += sender_to_receiver_count

    assert total_data_moved > 0


def create_shared_workspace(world_size, device_id=0):
    """Create a workspace tensor that can be accessed from all GPUs."""
    workspace_size = comm.get_moe_commworkspace_size_per_rank(world_size)
    torch.cuda.set_device(device_id)
    workspace = torch.zeros(
        world_size, workspace_size, dtype=torch.uint64, device=f"cuda:{device_id}"
    )
    return workspace


@requires_gpus(2)
@pytest.mark.parametrize(
    "world_size,input_entry_per_rank,vector_dim,dtype", CROSS_GPU_PARAMS
)
def test_cross_gpu_alltoall(world_size, input_entry_per_rank, vector_dim, dtype):
    """Test real cross-GPU alltoall communication with multiple ranks on different GPUs."""
    available_gpus = get_available_gpu_count()
    if world_size > available_gpus:
        pytest.skip(
            f"Test requires {world_size} GPUs, but only {available_gpus} available"
        )

    # Ensure we can access peer GPUs
    for i in range(world_size):
        for j in range(world_size):
            if i != j:
                torch.cuda.set_device(i)
                try:
                    # Test if we can access peer GPU memory
                    test_tensor = torch.tensor([1.0], device=f"cuda:{j}")
                    _ = test_tensor.cpu()  # Try to read from peer GPU
                except Exception as e:
                    pytest.skip(f"Cannot access GPU {j} from GPU {i}: {e}")

    # Create shared workspace accessible from all GPUs
    shared_workspace = create_shared_workspace(world_size, device_id=0)

    # Generate random target ranks for each rank - this creates variable communication patterns
    torch.manual_seed(0x12345)  # For reproducible test

    input_tensors_per_rank = []
    target_rank_ids_per_rank = []

    # Create input data on each GPU for each rank
    for rank in range(world_size):
        gpu_id = rank  # Map rank to GPU directly
        torch.cuda.set_device(gpu_id)

        # Create unique input data for this rank on this GPU
        input_tensor = (
            torch.randn(
                input_entry_per_rank, vector_dim, dtype=dtype, device=f"cuda:{gpu_id}"
            )
            + rank * 100
        )  # Add rank offset to make data distinguishable

        # Generate target ranks for this rank's data - creates alltoallv pattern
        target_rank_ids = torch.randint(
            0,
            world_size,
            (input_entry_per_rank,),
            dtype=torch.int32,
            device=f"cuda:{gpu_id}",
        )

        input_tensors_per_rank.append(input_tensor)
        target_rank_ids_per_rank.append(target_rank_ids)

    # Compute send patterns for each rank
    send_counts_per_rank = []
    send_cumsum_per_rank = []
    send_indices_per_rank = []

    for rank in range(world_size):
        gpu_id = rank
        torch.cuda.set_device(gpu_id)

        target_rank_ids = target_rank_ids_per_rank[rank]

        # Count how many vectors this rank sends to each other rank
        send_counts = torch.zeros(
            world_size, dtype=torch.int32, device=f"cuda:{gpu_id}"
        )
        for target_rank in range(world_size):
            mask = target_rank_ids == target_rank
            send_counts[target_rank] = mask.sum()

        send_cumsum = torch.cumsum(send_counts, dim=0).to(torch.int32)

        # Create send indices by sorting target rank IDs
        sorted_target_ranks, send_indices = torch.sort(target_rank_ids)
        send_indices = send_indices.to(torch.int32)

        send_counts_per_rank.append(send_counts)
        send_cumsum_per_rank.append(send_cumsum)
        send_indices_per_rank.append(send_indices)

    # Compute receive patterns for each rank
    recv_counts_per_rank = []
    recv_cumsum_per_rank = []
    recv_indices_per_rank = []
    output_tensors_per_rank = []

    for rank in range(world_size):
        gpu_id = rank
        torch.cuda.set_device(gpu_id)

        # Count how many vectors this rank receives from each other rank
        recv_counts = torch.zeros(
            world_size, dtype=torch.int32, device=f"cuda:{gpu_id}"
        )
        for sender_rank in range(world_size):
            recv_counts[sender_rank] = send_counts_per_rank[sender_rank][rank]

        recv_cumsum = torch.cumsum(recv_counts, dim=0).to(torch.int32)
        total_recv = recv_cumsum[-1].item()

        # Create output tensor for received data
        output_tensor = torch.zeros(
            total_recv, vector_dim, dtype=dtype, device=f"cuda:{gpu_id}"
        )

        # Simple receive indices (sequential)
        recv_indices = torch.arange(
            total_recv, dtype=torch.int32, device=f"cuda:{gpu_id}"
        )

        recv_counts_per_rank.append(recv_counts)
        recv_cumsum_per_rank.append(recv_cumsum)
        recv_indices_per_rank.append(recv_indices)
        output_tensors_per_rank.append(output_tensor)

    verify_cross_gpu_data_movement(
        input_tensors_per_rank,
        output_tensors_per_rank,
        send_counts_per_rank,
        recv_counts_per_rank,
        world_size,
    )

    # Create reference output by CPU simulation
    ref_output_tensors_per_rank = []

    for receiver_rank in range(world_size):
        gpu_id = receiver_rank
        torch.cuda.set_device(gpu_id)

        total_recv = recv_cumsum_per_rank[receiver_rank][-1].item()
        ref_output = torch.zeros(
            total_recv, vector_dim, dtype=dtype, device=f"cuda:{gpu_id}"
        )

        output_offset = 0
        for sender_rank in range(world_size):
            recv_count = recv_counts_per_rank[receiver_rank][sender_rank].item()
            if recv_count == 0:
                continue

            send_count = send_counts_per_rank[sender_rank][receiver_rank].item()
            assert (
                recv_count == send_count
            ), f"Count mismatch: {recv_count} != {send_count}"

            # Find which input vectors the sender is sending to this receiver
            target_rank_ids = target_rank_ids_per_rank[sender_rank]
            sender_to_receiver_mask = target_rank_ids == receiver_rank
            sender_to_receiver_indices = torch.where(sender_to_receiver_mask)[0]

            # Copy the data (move to same device first)
            sender_input = input_tensors_per_rank[sender_rank]
            if sender_input.device.index != gpu_id:
                sender_data = sender_input[sender_to_receiver_indices].to(
                    f"cuda:{gpu_id}"
                )
            else:
                sender_data = sender_input[sender_to_receiver_indices]

            ref_output[output_offset : output_offset + recv_count] = sender_data
            output_offset += recv_count

        ref_output_tensors_per_rank.append(ref_output)

    # Execute the actual cross-GPU MOE communication
    # Create streams for parallel execution
    streams = [torch.cuda.Stream(device=f"cuda:{rank}") for rank in range(world_size)]

    # Launch communication on all ranks in parallel
    for rank in range(world_size):
        gpu_id = rank
        with torch.cuda.device(gpu_id), torch.cuda.stream(streams[rank]):
            comm.moe_comm(
                input_tensors_per_rank[rank],
                send_cumsum_per_rank[rank],
                send_indices_per_rank[rank],
                output_tensors_per_rank[rank],
                recv_cumsum_per_rank[rank],
                recv_indices_per_rank[rank],
                shared_workspace,  # Same workspace shared across all ranks
                rank,  # This rank
                world_size,  # Total number of ranks
            )

    # Synchronize all streams
    for stream in streams:
        stream.synchronize()

    # Verify communication results
    for rank in range(world_size):
        gpu_id = rank
        torch.cuda.set_device(gpu_id)

        output_tensor = output_tensors_per_rank[rank]
        ref_output = ref_output_tensors_per_rank[rank]

        # Verify shapes match
        assert (
            output_tensor.shape == ref_output.shape
        ), f"Shape mismatch for rank {rank}: {output_tensor.shape} vs {ref_output.shape}"

        # Verify data matches
        torch.testing.assert_close(
            output_tensor,
            ref_output,
            rtol=1e-3,
            atol=1e-4,
            msg=f"Data mismatch for rank {rank} on GPU {gpu_id}",
        )

    # Additional verification: ensure data actually moved between GPUs
    cross_gpu_data_found = False
    for rank in range(world_size):
        gpu_id = rank
        output_data = output_tensors_per_rank[rank].cpu()

        # Check if this rank received data from other GPUs
        for sender_rank in range(world_size):
            if sender_rank == rank:
                continue

            recv_count = recv_counts_per_rank[rank][sender_rank].item()
            if recv_count > 0:
                # This rank received data from a different GPU
                cross_gpu_data_found = True
                sender_gpu = sender_rank

                break

    assert cross_gpu_data_found, "No evidence of actual cross-GPU data transfer found!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
