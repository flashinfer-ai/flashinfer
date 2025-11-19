# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import traceback

import pytest
import torch
from mpi4py import MPI

from flashinfer.comm import MoeAlltoAll
from flashinfer.comm.mapping import Mapping


@pytest.fixture(autouse=True)
def setup_test():
    torch.manual_seed(0x1234)


def compute_target_rank_id(expert_id, num_experts_per_rank):
    """Compute the rank that owns a given expert using contiguous partitioning."""
    return expert_id // num_experts_per_rank


def generate_token_selected_experts(
    local_num_tokens: int, ep_size: int, num_experts_per_rank: int, top_k: int
) -> torch.Tensor:
    """Generate global expert IDs tensor."""
    return torch.randint(
        0,
        ep_size * num_experts_per_rank,
        (local_num_tokens, top_k),
        dtype=torch.int32,
        device="cuda",
    )


def create_experts(
    num_experts_per_rank, hidden_size, ep_rank, device, dtype=torch.bfloat16
):
    """
    Create a 3D tensor of expert weights for a given rank.

    Returns:
        experts: Tensor of shape [num_experts_per_rank, hidden_size, hidden_size]
    """
    experts = torch.empty(
        (num_experts_per_rank, hidden_size, hidden_size), dtype=dtype, device=device
    )
    for i in range(num_experts_per_rank):
        torch.manual_seed(ep_rank * 1000 + i)
        torch.nn.init.xavier_uniform_(experts[i])
    return experts


def fake_moe(
    hidden_states,
    token_selected_experts,
    token_final_scales,
    experts,
    is_ep=False,
    ep_rank=None,
    num_experts_per_rank=None,
):
    """
    Emulate MoE computation.

    Returns:
        processed_states: [num_tokens, hidden_size]
    """
    num_tokens, _ = hidden_states.shape
    _, top_k = token_selected_experts.shape

    if is_ep:
        assert ep_rank is not None and num_experts_per_rank is not None

    processed_states = torch.zeros_like(hidden_states)

    for token_idx in range(num_tokens):
        for k in range(top_k):
            expert_id = token_selected_experts[token_idx, k].item()
            if is_ep:
                if not (
                    expert_id >= ep_rank * num_experts_per_rank
                    and expert_id < (ep_rank + 1) * num_experts_per_rank
                ):
                    continue
                local_expert_id = expert_id - ep_rank * num_experts_per_rank
                expert = experts[local_expert_id]
            else:
                expert = experts[expert_id]

            scale = token_final_scales[token_idx, k]
            processed_states[token_idx] += hidden_states[token_idx] @ expert * scale

    return processed_states


def make_bfloat16_payloads(
    local_num_tokens: int,
    hidden_size: int,
    top_k: int,
    rank: int,
    token_selected_experts: torch.Tensor,
) -> tuple[list, int]:
    """Create bfloat16 test payloads."""
    payloads = []

    # Payload 0: Hidden states
    hidden_states = torch.randn(
        local_num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    hidden_states += rank  # Add rank offset for verification
    payloads.append(hidden_states)

    # Payload 1: token_selected_experts
    payloads.append(token_selected_experts)

    # Payload 2: token_final_scales
    token_final_scales = torch.rand(
        local_num_tokens, top_k, dtype=torch.bfloat16, device="cuda"
    )
    payloads.append(token_final_scales)

    return payloads, 1  # expert_id_payload_index = 1


def run_moe_a2a_dispatch_single_rank(
    ep_size, all_num_tokens, top_k, num_experts_per_rank, hidden_size
):
    """Test MoE A2A dispatch on a single rank."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    if world_size != ep_size:
        pytest.skip(f"Test requires exactly {ep_size} ranks")

    torch.cuda.set_device(rank)

    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        gpus_per_node=world_size,
        tp_size=world_size,
        pp_size=1,
        cp_size=1,
    )

    local_num_tokens = all_num_tokens[rank]
    max_num_tokens = max(all_num_tokens)

    # Generate inputs
    token_selected_experts = generate_token_selected_experts(
        local_num_tokens, ep_size, num_experts_per_rank, top_k
    )

    payloads, expert_id_payload_index = make_bfloat16_payloads(
        local_num_tokens, hidden_size, top_k, rank, token_selected_experts
    )

    # Initialize MoeAlltoAll
    moe_a2a = MoeAlltoAll(
        mapping=mapping,
        max_num_tokens=max_num_tokens,
        top_k=top_k,
        num_experts=ep_size * num_experts_per_rank,
        workspace_size_per_rank=512 * 1024 * 1024,
    )

    # Dispatch
    recv_tensors = moe_a2a.dispatch(
        token_selected_experts=token_selected_experts,
        input_payloads=payloads,
        runtime_max_tokens_per_rank=max_num_tokens,
    )

    # Verify shapes
    assert len(recv_tensors) == len(payloads)
    for i, recv_tensor in enumerate(recv_tensors):
        assert recv_tensor.shape[0] == ep_size
        assert recv_tensor.shape[1] == max_num_tokens
        assert recv_tensor.shape[2] == payloads[i].shape[1]


def run_moe_a2a_dispatch_moe_combine_single_rank(
    ep_size, all_num_tokens, top_k, num_experts_per_rank, hidden_size
):
    """Test full MoE A2A dispatch + expert processing + combine cycle."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    if world_size != ep_size:
        pytest.skip(f"Test requires exactly {ep_size} ranks")

    torch.cuda.set_device(rank)

    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        gpus_per_node=world_size,
        tp_size=world_size,
        pp_size=1,
        cp_size=1,
    )

    local_num_tokens = all_num_tokens[rank]
    max_num_tokens = max(all_num_tokens)

    # Generate inputs
    token_selected_experts = generate_token_selected_experts(
        local_num_tokens, ep_size, num_experts_per_rank, top_k
    )

    payloads, expert_id_payload_index = make_bfloat16_payloads(
        local_num_tokens, hidden_size, top_k, rank, token_selected_experts
    )

    hidden_states = payloads[0]
    token_final_scales = payloads[2]

    # Create experts for this rank
    experts = create_experts(
        num_experts_per_rank, hidden_size, rank, "cuda", dtype=torch.bfloat16
    )

    # Compute reference (single-GPU MoE)
    all_experts = torch.cat(
        [
            create_experts(
                num_experts_per_rank, hidden_size, r, "cuda", dtype=torch.bfloat16
            )
            for r in range(ep_size)
        ],
        dim=0,
    )
    reference_output = fake_moe(
        hidden_states,
        token_selected_experts,
        token_final_scales,
        all_experts,
        is_ep=False,
    )

    # Initialize MoeAlltoAll
    moe_a2a = MoeAlltoAll(
        mapping=mapping,
        max_num_tokens=max_num_tokens,
        top_k=top_k,
        num_experts=ep_size * num_experts_per_rank,
        workspace_size_per_rank=512 * 1024 * 1024,
    )

    # Dispatch
    recv_tensors = moe_a2a.dispatch(
        token_selected_experts=token_selected_experts,
        input_payloads=payloads,
        runtime_max_tokens_per_rank=max_num_tokens,
    )

    # Unpack received tensors
    hidden_states_recv = recv_tensors[0]  # [ep_size, max_tokens, hidden_size]
    token_selected_experts_recv = recv_tensors[1]  # [ep_size, max_tokens, top_k]
    token_final_scales_recv = recv_tensors[2]  # [ep_size, max_tokens, top_k]

    # Get workspace-backed tensor for output
    moe_output = moe_a2a.get_combine_payload_tensor_in_workspace(
        runtime_max_tokens_per_rank=max_num_tokens,
        hidden_size=hidden_size,
        dtype=torch.bfloat16,
    )
    moe_output.zero_()

    # Process each rank's tokens with local experts
    for source_rank in range(ep_size):
        source_num_tokens = all_num_tokens[source_rank]
        for token_idx in range(source_num_tokens):
            for k in range(top_k):
                expert_id = token_selected_experts_recv[
                    source_rank, token_idx, k
                ].item()
                local_expert_id = expert_id - rank * num_experts_per_rank

                if 0 <= local_expert_id < num_experts_per_rank:
                    token_hidden = hidden_states_recv[source_rank, token_idx]
                    scale = token_final_scales_recv[source_rank, token_idx, k]
                    expert_out = token_hidden @ experts[local_expert_id]
                    output_idx = source_rank * max_num_tokens + token_idx
                    moe_output[output_idx] += expert_out * scale

    # Combine
    combined_output = moe_a2a.combine(
        payload=moe_output.view(ep_size, max_num_tokens, hidden_size),
        runtime_max_tokens_per_rank=max_num_tokens,
        payload_in_workspace=True,
    )

    # Verify against reference
    torch.testing.assert_close(combined_output, reference_output, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("ep_size", [2, 4])
@pytest.mark.parametrize("all_num_tokens", [[64, 64], [32, 48, 64, 80]])
@pytest.mark.parametrize("top_k", [2, 4])
@pytest.mark.parametrize("num_experts_per_rank", [2, 4])
@pytest.mark.parametrize("hidden_size", [128, 256])
def test_moe_a2a_dispatch(
    ep_size, all_num_tokens, top_k, num_experts_per_rank, hidden_size
):
    """Test MoE A2A dispatch operation."""
    if len(all_num_tokens) != ep_size:
        pytest.skip(
            f"all_num_tokens length {len(all_num_tokens)} must match ep_size {ep_size}"
        )

    try:
        run_moe_a2a_dispatch_single_rank(
            ep_size, all_num_tokens, top_k, num_experts_per_rank, hidden_size
        )
    except Exception as e:
        traceback.print_exc()
        raise e


@pytest.mark.parametrize("ep_size", [2, 4])
@pytest.mark.parametrize("all_num_tokens", [[64, 64], [32, 48, 64, 80]])
@pytest.mark.parametrize("top_k", [2, 4])
@pytest.mark.parametrize("num_experts_per_rank", [2, 4])
@pytest.mark.parametrize("hidden_size", [128, 256])
def test_moe_a2a_dispatch_moe_combine(
    ep_size, all_num_tokens, top_k, num_experts_per_rank, hidden_size
):
    """Test full MoE A2A dispatch + expert processing + combine cycle."""
    if len(all_num_tokens) != ep_size:
        pytest.skip(
            f"all_num_tokens length {len(all_num_tokens)} must match ep_size {ep_size}"
        )

    try:
        run_moe_a2a_dispatch_moe_combine_single_rank(
            ep_size, all_num_tokens, top_k, num_experts_per_rank, hidden_size
        )
    except Exception as e:
        traceback.print_exc()
        raise e


if __name__ == "__main__":
    # Run with: mpirun -n 2 python -m pytest tests/comm/test_mnnvl_a2a.py -v
    pytest.main([__file__, "-v", "-s"])
