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

from flashinfer.comm.mapping import Mapping

import flashinfer.comm.trtllm_moe_alltoall as trtllm_moe_alltoall


@pytest.fixture(autouse=True, scope="session")
def setup_test_environment():
    """Set up test environment and warm up JIT compilation."""
    torch.manual_seed(0xD5)
    yield


# Single GPU test parameters
SINGLE_GPU_PARAMS = [
    (902, 32768, 256, 8, torch.float16),  # Large data, float16
    (101, 288, 128, 4, torch.float16),  # Medium data, float16
    (902, 7168, 256, 8, torch.bfloat16),  # Large data, bfloat16
    (101, 288, 128, 4, torch.bfloat16),  # Medium data, bfloat16
    (10, 8, 8, 2, torch.bfloat16),  # Small data, bfloat16
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
    "num_tokens,vector_dim,num_experts,top_k,dtype",
    SINGLE_GPU_PARAMS,
)
def test_moe_alltoall_single_gpu(num_tokens, vector_dim, num_experts, top_k, dtype):
    """Test MOE alltoall communication on single GPU."""
    torch.cuda.set_device(0)
    # Create a random input tensor
    input_tensor1 = torch.randn(
        num_tokens, vector_dim, dtype=dtype, device=torch.device("cuda")
    )
    input_tensor2 = torch.randn(
        num_tokens, vector_dim * 2, dtype=dtype, device=torch.device("cuda")
    )

    token_selected_experts = torch.empty(
        num_tokens, top_k, dtype=torch.int32, device=torch.device("cuda")
    )
    for i in range(num_tokens):
        # Include one extra expert to represent invalid expert IDs
        token_selected_experts[i] = torch.randperm(
            num_experts, dtype=torch.int32, device=torch.device("cuda")
        )[:top_k]
    token_selected_experts = token_selected_experts.contiguous()

    workspace_size = trtllm_moe_alltoall.moe_a2a_get_workspace_size_per_rank(
        1,
        num_tokens,
        input_tensor1.numel() * dtype.itemsize
        + input_tensor2.numel() * dtype.itemsize
        + token_selected_experts.numel() * torch.int32.itemsize,
    )
    mapping = Mapping(rank=0, world_size=1)
    moe_a2a = trtllm_moe_alltoall.MoeAlltoAll(
        mapping,
        num_tokens,
        top_k,
        num_experts,
        workspace_size_per_rank=workspace_size,
    )

    output_tensor1, output_tensor2, token_selected_experts_output = moe_a2a.dispatch(
        token_selected_experts,
        [input_tensor1, input_tensor2, token_selected_experts],
        num_tokens,
        invalid_token_expert_id=-3,  # Tokens assigned to invalid expert are set to -3
        expert_id_payload_index=2,
    )

    # Sort to undo the shuffling that happens in the dispatch kernel.
    input_tensor1, _ = torch.sort(input_tensor1, dim=0)
    input_tensor2, _ = torch.sort(input_tensor2, dim=0)
    token_selected_experts, _ = torch.sort(token_selected_experts, dim=0)
    output_tensor1, _ = torch.sort(output_tensor1[0], dim=0)
    output_tensor2, _ = torch.sort(output_tensor2[0], dim=0)
    token_selected_experts_output, _ = torch.sort(
        token_selected_experts_output[0], dim=0
    )

    torch.testing.assert_close(output_tensor1, input_tensor1, atol=0, rtol=0)
    torch.testing.assert_close(output_tensor2, input_tensor2, atol=0, rtol=0)
    torch.testing.assert_close(
        token_selected_experts_output, token_selected_experts, atol=0, rtol=0
    )

    moe_a2a._reset_workspace()


@pytest.mark.parametrize("world_size,num_tokens,vector_dim,dtype", MULTI_RANK_PARAMS)
def test_moe_alltoall_multi_rank_single_gpu(world_size, num_tokens, vector_dim, dtype):
    """Test MOE alltoall communication with multiple ranks on single GPU."""
    torch.cuda.set_device(0)
    max_world_size = 8
    assert world_size <= max_world_size, (
        f"should run with world_size at most {max_world_size}"
    )

    # SM count is now set up globally in the fixture

    # Create a random input tensor
    input_tensors = [
        torch.randn(
            num_tokens * world_size,
            vector_dim * (i + 1),
            dtype=dtype,
            device=torch.device("cuda"),
        )
        for i in range(2)
    ]

    token_selected_experts = torch.randint(
        0,
        world_size,
        (num_tokens * world_size, 1),
        dtype=torch.int32,
        device=torch.device("cuda"),
    )

    payloads = input_tensors + [token_selected_experts]
    total_payload_size_per_element = [x[0].numel() * x.itemsize for x in payloads]
    total_payload_size_per_element = sum(total_payload_size_per_element)

    workspace_size = trtllm_moe_alltoall.moe_a2a_get_workspace_size_per_rank(
        world_size, num_tokens * world_size, total_payload_size_per_element
    )

    all_workspaces = torch.zeros(
        world_size, workspace_size, dtype=torch.uint8, device=torch.device("cuda")
    )

    # Must be done before the synchronization so the state is cleared
    metainfo = []
    for rank in range(world_size):
        metainfo.append(
            trtllm_moe_alltoall.moe_a2a_initialize(
                all_workspaces,
                rank,
                world_size,
                num_tokens * world_size,
            )
        )

    # Synchronize before starting parallel communication
    torch.cuda.synchronize()

    output_tensors = []
    # do alltoall in parallel
    cuda_streams_all_ranks = [torch.cuda.Stream() for _ in range(world_size)]
    for rank in range(world_size):
        with torch.cuda.stream(cuda_streams_all_ranks[rank]):
            rank_payloads = [
                x[rank * num_tokens : (rank + 1) * num_tokens] for x in payloads
            ]
            output_tensors.append(
                trtllm_moe_alltoall.moe_a2a_dispatch(
                    rank_payloads[2],
                    rank_payloads,
                    all_workspaces,
                    metainfo[rank],
                    num_tokens,
                    ep_rank=rank,
                    ep_size=world_size,
                    top_k=1,
                    num_experts=world_size,
                )[0]
            )

    for rank in range(world_size):
        cuda_streams_all_ranks[rank].synchronize()

    torch.cuda.synchronize()

    torch.set_printoptions(threshold=float("inf"))
    print(
        f"all_workspaces: {all_workspaces.shape} {all_workspaces.flatten().view(torch.uint8)[1152:1632].view(torch.bfloat16)}"
    )

    for rank in range(world_size):
        print(f"output_tensors[{rank}]: {output_tensors[rank]}")

    for rank in range(world_size):
        # Get the indices where token_selected_experts == rank
        print(
            f"token_selected_experts: {token_selected_experts.shape} {token_selected_experts}"
        )
        token_selected_experts_indices = (
            token_selected_experts.flatten() == rank
        ).nonzero(as_tuple=False)

        for actual, ref in zip(output_tensors[rank], payloads, strict=True):
            print(f"token_selected_experts_indices: {token_selected_experts_indices}")
            print(f"actual raw: {actual.shape} {actual}")
            actual = actual[rank][: len(token_selected_experts_indices)]
            print(f"actual filtered: {actual.shape} {actual}")
            ref = ref[token_selected_experts_indices].squeeze()
            actual, _ = torch.sort(actual, dim=0)
            ref, _ = torch.sort(ref, dim=0)
            print(f"actual: {actual}")
            print(f"ref: {ref}")
            torch.testing.assert_close(actual, ref, atol=0, rtol=0)


# TODO Add a combine test

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
