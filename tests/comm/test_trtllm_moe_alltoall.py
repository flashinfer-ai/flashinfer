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
    (902, 7168, 256, 8),  # Large data
    (10, 288, 128, 4),  # Medium data
    (10, 8, 8, 2),  # Small data
]

MULTI_RANK_PARAMS = [
    (2, 5, 8),  # Small input, 2 ranks
    (4, 901, 32768),  # Large input, 4 ranks
    (8, 16384, 128),  # Many small vectors, 8 ranks
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


def make_payload(num_tokens, vector_dim, dtype):
    if dtype == torch.uint8 or dtype == torch.int32:
        return torch.randint(
            torch.iinfo(dtype).min,
            torch.iinfo(dtype).max,
            (num_tokens, vector_dim),
            dtype=dtype,
            device=torch.device("cuda"),
        )
    else:
        return torch.randn(
            num_tokens, vector_dim, dtype=dtype, device=torch.device("cuda")
        )


@pytest.mark.parametrize(
    "num_tokens,vector_dim,num_experts,top_k",
    SINGLE_GPU_PARAMS,
)
def test_moe_alltoall_single_gpu(num_tokens, vector_dim, num_experts, top_k):
    """Test MOE alltoall communication on single GPU."""
    torch.cuda.set_device(0)
    # Create a random input tensor
    dtypes = [torch.float16, torch.bfloat16, torch.int32, torch.uint8]
    input_tensors = [
        make_payload(num_tokens, vector_dim * (i + 1), dtype)
        for i, dtype in enumerate(dtypes)
    ]

    token_selected_experts = torch.empty(
        num_tokens, top_k, dtype=torch.int32, device=torch.device("cuda")
    )
    for i in range(num_tokens):
        # Include one extra expert to represent invalid expert IDs
        token_selected_experts[i] = torch.randperm(
            num_experts, dtype=torch.int32, device=torch.device("cuda")
        )[:top_k]
    token_selected_experts = token_selected_experts.contiguous()

    payload_size_per_token = sum([x[0].numel() * x.itemsize for x in input_tensors])

    workspace_size = trtllm_moe_alltoall.moe_a2a_get_workspace_size_per_rank(
        1,
        num_tokens,
        payload_size_per_token,
    )
    mapping = Mapping(rank=0, world_size=1)
    moe_a2a = trtllm_moe_alltoall.MoeAlltoAll(
        mapping,
        num_tokens,
        top_k,
        num_experts,
        workspace_size_per_rank=workspace_size,
    )

    output_tensors = moe_a2a.dispatch(
        token_selected_experts,
        input_tensors,
        num_tokens,
        invalid_token_expert_id=-3,  # Tokens assigned to invalid expert are set to -3
        expert_id_payload_index=2,
    )

    # Sort to undo the shuffling that happens in the dispatch kernel.
    for input_tensor, output_tensor in zip(input_tensors, output_tensors, strict=True):
        input_tensor, _ = torch.sort(input_tensor, dim=0)
        output_tensor, _ = torch.sort(output_tensor.flatten(end_dim=1), dim=0)
        torch.testing.assert_close(output_tensor, input_tensor, atol=0, rtol=0)

    moe_a2a._reset_workspace()


@pytest.mark.parametrize("world_size,num_tokens,vector_dim", MULTI_RANK_PARAMS)
def test_moe_alltoall_multi_rank_single_gpu(world_size, num_tokens, vector_dim):
    """Test MOE alltoall communication with multiple ranks on single GPU."""
    torch.cuda.set_device(0)
    max_world_size = 8
    assert world_size <= max_world_size, (
        f"should run with world_size at most {max_world_size}"
    )

    dtypes = [torch.float16, torch.bfloat16, torch.int32, torch.uint8]
    # Create a random input tensor
    input_tensors = [
        make_payload(num_tokens * world_size, vector_dim * (i + 1), dtype)
        for i, dtype in enumerate(dtypes)
    ]

    token_selected_experts = torch.randint(
        0,
        world_size,
        (num_tokens * world_size, 1),
        dtype=torch.int32,
        device=torch.device("cuda"),
    )

    payloads = input_tensors
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
            rank_token_selected_experts = token_selected_experts[
                rank * num_tokens : (rank + 1) * num_tokens
            ]
            output_tensors.append(
                trtllm_moe_alltoall.moe_a2a_dispatch(
                    rank_token_selected_experts,
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

    for rank in range(world_size):
        # Get the indices where token_selected_experts == rank
        token_selected_experts_indices = (
            token_selected_experts.flatten() == rank
        ).nonzero(as_tuple=False)

        for actual, ref in zip(output_tensors[rank][:-1], payloads[:-1], strict=True):
            # Select the tensors that arent all zeros
            actual = actual.flatten(end_dim=1)
            actual = actual[actual.any(dim=1)]
            ref = ref[token_selected_experts_indices].squeeze()
            actual, _ = torch.sort(actual, dim=0)
            ref, _ = torch.sort(ref, dim=0)
            torch.testing.assert_close(actual, ref, atol=0, rtol=0)


# TODO Add a combine test

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
