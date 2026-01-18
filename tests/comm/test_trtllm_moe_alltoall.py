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
import pynvml
from flashinfer.comm.mapping import Mapping

import flashinfer.comm.trtllm_moe_alltoall as trtllm_moe_alltoall

from .conftest import mnnvl_available

pynvml.nvmlInit()


@pytest.fixture(autouse=True, scope="session")
def setup_test_environment():
    """Set up torch seed for deterministic tests."""
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
    (4, 32, 32768),  # Large input, 4 ranks
    (8, 16, 2048),  # Medium input, 8 ranks
]

SANITIZE_PARAMS = [
    (2, 64),  # 2 ranks
    (4, 32),  # 4 ranks
    (8, 16),  # 8 ranks
]

COMBINE_PARAMS = [
    (2, 64, 8, 2, torch.bfloat16, True),  # Small input, 2 ranks
    (4, 32, 32768, 4, torch.bfloat16, True),  # Large input, 4 ranks
    (8, 16, 2048, 8, torch.bfloat16, True),  # Medium input, 8 ranks
    (8, 16, 2048, 8, torch.bfloat16, False),  # Medium input, 8 ranks
    (2, 64, 8, 2, torch.float16, True),  # Small input, 2 ranks
    (4, 32, 32768, 4, torch.float16, True),  # Large input, 4 ranks
    (8, 16, 2048, 8, torch.float16, True),  # Medium input, 8 ranks
    (8, 16, 2048, 8, torch.float16, False),  # Medium input, 8 ranks
]


# This is a hack to ensure we get forward progress when running multiple kernels on a single GPU
def check_sufficient_sm_count(num_tokens, world_size):
    if (
        num_tokens * world_size
        > torch.cuda.get_device_properties(0).multi_processor_count
    ):
        pytest.skip(
            f"Requires at least {num_tokens * world_size} SMs, but only {torch.cuda.get_device_properties(0).multi_processor_count} available"
        )


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
@pytest.mark.skipif(
    not mnnvl_available(),
    reason="Mnnvl memory is not supported on this platform or container lacks SYS_PTRACE capability",
)
def test_moe_alltoall_single_gpu(num_tokens, vector_dim, num_experts, top_k):
    """Test MOE alltoall communication on single GPU."""
    torch.cuda.set_device(0)
    # Create a random input tensor
    dtypes = [torch.bfloat16, torch.float16, torch.int32, torch.uint8]
    hidden_state_index = 0
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
        input_tensors[0].shape[-1] * input_tensors[0].itemsize,
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

    inplace_combine_tensor = moe_a2a.get_combine_payload_tensor_in_workspace(
        num_tokens,
        input_tensors[hidden_state_index].shape[-1],
        input_tensors[hidden_state_index].dtype,
    )

    # Copy first output tensor into inplace_combine_tensor
    inplace_combine_tensor.copy_(output_tensors[hidden_state_index])

    output = moe_a2a.combine(
        inplace_combine_tensor, num_tokens, payload_in_workspace=True
    )

    # Should just be a direct copy for 1 GPU
    torch.testing.assert_close(
        output, input_tensors[hidden_state_index], atol=0, rtol=0
    )


def dispatch_from_single_rank(
    input_tensors,
    token_selected_experts,
    world_size,
    num_experts,
    num_tokens,
    hidden_state_index=None,
):
    payloads = input_tensors
    total_payload_size_per_element = [x[0].numel() * x.itemsize for x in payloads]
    total_payload_size_per_element = sum(total_payload_size_per_element)

    combine_size = 0
    if hidden_state_index is not None:
        combine_size = (
            input_tensors[hidden_state_index].shape[-1]
            * input_tensors[hidden_state_index].itemsize
        )

    workspace_size = trtllm_moe_alltoall.moe_a2a_get_workspace_size_per_rank(
        world_size,
        num_tokens * world_size,
        total_payload_size_per_element,
        combine_size,
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
    combine_payload_offsets = []
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
            output, offset = trtllm_moe_alltoall.moe_a2a_dispatch(
                rank_token_selected_experts,
                rank_payloads,
                all_workspaces,
                metainfo[rank],
                num_tokens,
                ep_rank=rank,
                ep_size=world_size,
                top_k=rank_token_selected_experts.shape[-1],
                num_experts=num_experts,
            )
            output_tensors.append(output)
            combine_payload_offsets.append(offset)

    for rank in range(world_size):
        cuda_streams_all_ranks[rank].synchronize()

    torch.cuda.synchronize()

    return output_tensors, all_workspaces, metainfo, combine_payload_offsets


def sanitize_expert_ids_from_single_rank(
    output_tensors,
    expert_ids_index,
    all_workspaces,
    metainfo,
    world_size,
    invalid_expert_id,
):
    for rank in range(world_size):
        trtllm_moe_alltoall.moe_a2a_sanitize_expert_ids(
            output_tensors[rank][expert_ids_index],
            all_workspaces,
            metainfo[rank],
            rank,
            invalid_expert_id,
        )
    return output_tensors


def combine_from_single_rank(
    combine_payload,
    num_tokens,
    top_k,
    all_workspaces,
    metainfo,
    world_size,
    combine_payload_offsets,
    payload_in_workspace,
):
    combine_results = []

    torch.cuda.synchronize()

    cuda_streams_all_ranks = [torch.cuda.Stream() for _ in range(world_size)]
    for rank in range(world_size):
        with torch.cuda.stream(cuda_streams_all_ranks[rank]):
            combine_results.append(
                trtllm_moe_alltoall.moe_a2a_combine(
                    combine_payload[rank],
                    num_tokens,
                    all_workspaces,
                    metainfo[rank],
                    num_tokens,
                    ep_rank=rank,
                    ep_size=world_size,
                    top_k=top_k,
                    combine_payload_offset=combine_payload_offsets[rank],
                    payload_in_workspace=payload_in_workspace,
                )
            )

    for rank in range(world_size):
        cuda_streams_all_ranks[rank].synchronize()

    torch.cuda.synchronize()

    return combine_results


@pytest.mark.parametrize("world_size,num_tokens,vector_dim", MULTI_RANK_PARAMS)
def test_moe_alltoall_multi_rank_single_gpu(world_size, num_tokens, vector_dim):
    """Test MOE alltoall communication with multiple ranks on single GPU."""
    torch.cuda.set_device(0)
    check_sufficient_sm_count(num_tokens, world_size)
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

    output_tensors, _, _, _ = dispatch_from_single_rank(
        input_tensors, token_selected_experts, world_size, world_size, num_tokens
    )

    for rank in range(world_size):
        # Get the indices where token_selected_experts == rank
        token_selected_experts_indices = (
            token_selected_experts.flatten() == rank
        ).nonzero(as_tuple=False)

        for actual, ref in zip(output_tensors[rank], input_tensors, strict=True):
            # Select the tensors that arent all zeros
            actual = actual.flatten(end_dim=1)
            actual = actual[actual.any(dim=1)]
            ref = ref[token_selected_experts_indices].squeeze()
            actual, _ = torch.sort(actual, dim=0)
            ref, _ = torch.sort(ref, dim=0)
            torch.testing.assert_close(actual, ref, atol=0, rtol=0)


@pytest.mark.parametrize("world_size,num_tokens", SANITIZE_PARAMS)
def test_sanitize_expert_ids(world_size, num_tokens):
    torch.cuda.set_device(0)
    check_sufficient_sm_count(num_tokens, world_size)
    max_world_size = 8
    assert world_size <= max_world_size, (
        f"should run with world_size at most {max_world_size}"
    )

    flags = torch.ones(
        num_tokens * world_size, 1, dtype=torch.bool, device=torch.device("cuda")
    )
    token_selected_experts = torch.randint(
        0,
        world_size,
        (num_tokens * world_size, 1),
        dtype=torch.int32,
        device=torch.device("cuda"),
    )

    output_tensors, all_workspaces, metainfo, _ = dispatch_from_single_rank(
        [token_selected_experts, flags],
        token_selected_experts,
        world_size,
        world_size,
        num_tokens,
    )

    # Clone since the tensors are modified in place
    expected_output_tensors = [(x[0].clone(), x[1].clone()) for x in output_tensors]
    output_tensors = sanitize_expert_ids_from_single_rank(
        output_tensors, 0, all_workspaces, metainfo, world_size, -3
    )

    for rank, (sanitized, raw) in enumerate(
        zip(output_tensors, expected_output_tensors, strict=True)
    ):
        raw_tensor, flag_tensor = raw
        valid_mask = (raw_tensor == rank) & flag_tensor
        raw_tensor[~valid_mask] = -3
        torch.testing.assert_close(sanitized[0], raw_tensor, atol=0, rtol=0)


def fake_moe(
    hidden_states: torch.Tensor,
    token_selected_experts: torch.Tensor,
    num_experts: int,
    is_ep: bool = False,
    ep_rank: int | None = None,
    num_experts_per_rank: int | None = None,
) -> torch.Tensor:
    """
    Apply a deterministic fake MoE transformation for validation.

    Each expert applies a predictable scale: (expert_id + 1.0) / num_experts + 0.5
    This allows verifying that communication correctly routes tokens to experts
    and combines results.

    Args:
        hidden_states: Input tensor [num_tokens, hidden_size] or [world_size, num_tokens, hidden_size]
        token_selected_experts: Expert assignments [num_tokens, top_k] or [world_size, num_tokens, top_k]
        num_experts: Total number of experts
        is_ep: If True, only process experts assigned to this rank
        ep_rank: Rank for expert parallel filtering
        num_experts_per_rank: Number of experts per rank

    Returns:
        Processed tensor with same shape as hidden_states
    """
    target_shape = hidden_states.shape
    hidden_states = hidden_states.flatten(end_dim=-2)
    token_selected_experts = token_selected_experts.flatten(end_dim=-2)
    num_tokens, _ = hidden_states.shape
    _, top_k = token_selected_experts.shape

    if is_ep:
        assert ep_rank is not None and num_experts_per_rank is not None

    # Initialize output
    processed_states = torch.zeros_like(hidden_states)

    # Process each token
    for token_idx in range(num_tokens):
        results = []
        # For each expert selected for this token
        for k in range(top_k):
            expert_id = token_selected_experts[token_idx, k].item()
            if is_ep and not (
                expert_id >= ep_rank * num_experts_per_rank
                and expert_id < (ep_rank + 1) * num_experts_per_rank
            ):
                continue

            scale = (expert_id + 1.0) / num_experts + 0.5
            results.append(hidden_states[token_idx] * scale)

        # Summing the results after is closer to the actual implementation as we do a tree reduction.
        if results:
            processed_states[token_idx] = torch.sum(
                torch.stack(results, dim=0), dim=0, dtype=torch.float32
            ).to(processed_states.dtype)

    return processed_states.view(target_shape)


@pytest.mark.parametrize(
    "world_size,num_tokens,vector_dim,top_k,dtype,payload_in_workspace", COMBINE_PARAMS
)
def test_moe_combine_multi_rank_single_gpu(
    world_size, num_tokens, vector_dim, top_k, dtype, payload_in_workspace
):
    torch.cuda.set_device(0)
    check_sufficient_sm_count(num_tokens, world_size)
    max_world_size = 8
    assert world_size <= max_world_size, (
        f"should run with world_size at most {max_world_size}"
    )

    num_experts = world_size * top_k

    token_selected_experts_index = 0
    hidden_state_index = 1

    token_selected_experts = torch.empty(
        num_tokens * world_size, top_k, dtype=torch.int32, device=torch.device("cuda")
    )

    for i in range(num_tokens * world_size):
        # Include one extra expert to represent invalid expert IDs
        token_selected_experts[i] = torch.randperm(
            num_experts, dtype=torch.int32, device=torch.device("cuda")
        )[:top_k]
    token_selected_experts = token_selected_experts.contiguous()

    # Create a random input tensor
    reference_tensor = make_payload(num_tokens * world_size, vector_dim, dtype)
    input_tensors = [
        token_selected_experts,
        reference_tensor,
        make_payload(
            num_tokens * world_size, 1, torch.uint8
        ),  # Some extra payload to test combine alignment logic
    ]

    output_tensors, all_workspaces, metainfo, combine_payload_offsets = (
        dispatch_from_single_rank(
            input_tensors,
            token_selected_experts,
            world_size,
            num_experts,
            num_tokens,
            hidden_state_index,
        )
    )

    # Sanitize expert ids for fake_moe
    output_tensors = sanitize_expert_ids_from_single_rank(
        output_tensors,
        token_selected_experts_index,
        all_workspaces,
        metainfo,
        world_size,
        -1,
    )

    inplace_combine_tensors = []
    for rank in range(world_size):
        if payload_in_workspace:
            inplace_combine_tensors.append(
                trtllm_moe_alltoall.moe_a2a_wrap_payload_tensor_in_workspace(
                    all_workspaces[rank, :],
                    [world_size, num_tokens],
                    combine_payload_offsets[rank],
                    combine_payload_offsets[rank]
                    + world_size * num_tokens * vector_dim * dtype.itemsize,
                    dtype,
                )
            )
        else:
            inplace_combine_tensors.append(
                torch.empty(
                    world_size,
                    num_tokens,
                    vector_dim,
                    dtype=dtype,
                    device=torch.device("cuda"),
                )
            )

    for rank in range(world_size):
        inplace_combine_tensors[rank].copy_(
            fake_moe(
                output_tensors[rank][hidden_state_index],
                output_tensors[rank][token_selected_experts_index],
                num_experts,
                is_ep=True,
                ep_rank=rank,
                num_experts_per_rank=num_experts // world_size,
            )
        )

    combine_results = combine_from_single_rank(
        inplace_combine_tensors,
        num_tokens,
        top_k,
        all_workspaces,
        metainfo,
        world_size,
        combine_payload_offsets,
        payload_in_workspace=payload_in_workspace,
    )

    reference_result = fake_moe(
        input_tensors[hidden_state_index], token_selected_experts, num_experts
    )

    for rank in range(world_size):
        torch.testing.assert_close(
            combine_results[rank],
            reference_result[rank * num_tokens : (rank + 1) * num_tokens],
            atol=1.5e-2,
            rtol=1.5e-2,
        )


@pytest.mark.skipif(
    not mnnvl_available(),
    reason="Mnnvl memory is not supported on this platform or container lacks SYS_PTRACE capability",
)
def test_moe_workspace_size_per_rank():
    """Test the workspace size per rank for the MoeAlltoAll operation."""
    ep_size = 8
    num_tokens = 10
    hidden_size = 128
    topk = 2
    raw_workspace_size = trtllm_moe_alltoall.moe_a2a_get_workspace_size_per_rank(
        ep_size,
        num_tokens,
        (hidden_size * torch.bfloat16.itemsize + topk * 4 + topk * 4),
        hidden_size * torch.bfloat16.itemsize,
    )
    assert raw_workspace_size > 0

    moe_workspace_size = (
        trtllm_moe_alltoall.MoeAlltoAll.get_moe_workspace_size_per_rank(
            ep_size, topk, num_tokens, hidden_size
        )
    )
    assert moe_workspace_size == raw_workspace_size

    empty_workspace_size = trtllm_moe_alltoall.moe_a2a_get_workspace_size_per_rank(
        ep_size, num_tokens, 0, 0
    )

    assert empty_workspace_size > 0
    assert (
        empty_workspace_size
        == trtllm_moe_alltoall.get_moe_alltoall_module().moe_a2a_get_aux_data_size(
            ep_size, num_tokens
        )
    )

    non_empty_workspace_size = trtllm_moe_alltoall.moe_a2a_get_workspace_size_per_rank(
        ep_size, num_tokens, hidden_size, hidden_size
    )

    actual_data_size = non_empty_workspace_size - empty_workspace_size
    assert actual_data_size == hidden_size * ep_size * num_tokens * 2

    mapping = Mapping(rank=0, world_size=1)
    moe_a2a = trtllm_moe_alltoall.MoeAlltoAll(
        mapping, num_tokens, topk, ep_size, hidden_size=hidden_size
    )
    raw_workspace_size = (
        trtllm_moe_alltoall.MoeAlltoAll.get_moe_workspace_size_per_rank(
            mapping.moe_ep_size, topk, num_tokens, hidden_size
        )
    )
    assert moe_a2a.workspace_size_per_rank == raw_workspace_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
