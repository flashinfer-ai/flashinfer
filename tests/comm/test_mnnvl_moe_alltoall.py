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
from flashinfer.comm.mnnvl import MnnvlMemory

from .conftest import mnnvl_available


class MPIExit(Exception):
    pass


def check_any_rank_failed():
    comm = MPI.COMM_WORLD
    if any(comm.allgather(False)):
        raise MPIExit("Another rank failed")


def safe_run(func, *args, **kwargs):
    comm = MPI.COMM_WORLD
    try:
        func(*args, **kwargs)
    except MPIExit:
        raise
    except Exception:
        traceback.print_exc()
        comm.allgather(True)
        raise


@pytest.fixture(autouse=True)
def setup_test():
    torch.manual_seed(0x1234)
    yield


def compute_target_rank_id(expert_id, num_experts_per_rank):
    """Compute the rank that owns a given expert using contiguous partitioning.
    Experts are divided evenly across ranks:
      - Rank 0: experts [0, num_experts_per_rank)
      - Rank 1: experts [num_experts_per_rank, 2 * num_experts_per_rank)
      - ...
    For example, with 32 experts and 4 ranks (8 experts per rank):
      - Rank 0: experts 0-7
      - Rank 1: experts 8-15
      - Rank 2: experts 16-23
      - Rank 3: experts 24-31
    """
    return expert_id // num_experts_per_rank


def generate_token_selected_experts(
    local_num_tokens: int, ep_size: int, num_experts_per_rank: int, top_k: int
) -> torch.Tensor:
    """Generate global expert IDs tensor, aligned with single-GPU test semantics."""
    if local_num_tokens == 0:
        return torch.empty(0, top_k, dtype=torch.int32, device="cuda")

    # Select topk random experts for each token
    def select_experts(items, topk):
        perm = torch.randperm(items, dtype=torch.int32, device="cuda")
        return perm[:topk]

    return torch.stack(
        [
            select_experts(ep_size * num_experts_per_rank, top_k)
            for _ in range(local_num_tokens)
        ],
        dim=0,
    )


def create_experts(
    num_experts_per_rank, hidden_size, ep_rank, device, dtype=torch.bfloat16
):
    """
    Create a 3D tensor of expert weights for a given rank.

    Args:
        num_experts_per_rank: Number of experts on this rank
        hidden_size: Hidden dimension size
        ep_rank: EP rank ID
        device: Device to create experts on

    Returns:
        experts: Tensor of shape [num_experts_per_rank, hidden_size, hidden_size]
    """

    # A simpler to debug initialization
    # identity = torch.eye(hidden_size, dtype=dtype, device=device)
    # return torch.stack([identity * (i + 1) for i in range(num_experts_per_rank)], dim=0)

    # For reproducibility, set the seed based on rank
    experts = torch.empty(
        (num_experts_per_rank, hidden_size, hidden_size), dtype=dtype, device=device
    )
    for i in range(num_experts_per_rank):
        torch.manual_seed(ep_rank * 1000 + i)
        # Xavier uniform initialization for each expert
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
    Emulate MoE computation by scaling tokens based on which experts belong to this rank.

    Args:
        hidden_states: [num_tokens, hidden_size] - input hidden states
        token_selected_experts: [num_tokens, top_k] - selected expert indices
        token_final_scales: [num_tokens, top_k] - scaling factors for each expert
        experts: [num_experts_per_rank, hidden_size, hidden_size] if is_ep, otherwise [num_experts, hidden_size, hidden_size] - expert weights
        is_ep: If true, emulate MoE on a EP rank; otherwise, emulate MoE with all experts
        ep_rank: EP rank ID
        num_experts_per_rank: Number of experts per rank

    Returns:
        processed_states: [num_tokens, hidden_size] - processed hidden states
    """
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
            if is_ep:
                if not (
                    expert_id >= ep_rank * num_experts_per_rank
                    and expert_id < (ep_rank + 1) * num_experts_per_rank
                ):
                    continue
                # Convert global expert ID to local expert ID for this rank
                local_expert_id = expert_id - ep_rank * num_experts_per_rank
                expert = experts[local_expert_id]
            else:
                expert = experts[expert_id]

            scale = token_final_scales[token_idx, k]
            results.append(hidden_states[token_idx] @ expert * scale)

        # Summing the results after is closer to the actual implementation as we do a tree reduction.
        if results:
            processed_states[token_idx] = torch.sum(
                torch.stack(results, dim=0), dim=0, dtype=torch.float32
            ).to(processed_states.dtype)

    return processed_states


def make_nvfp4_payloads(
    local_num_tokens: int,
    hidden_size: int,
    top_k: int,
    rank: int,
    token_selected_experts: torch.Tensor,
) -> tuple[list, int]:
    """Create the four NV FP4 payloads exactly as in single-GPU test."""
    payloads = []
    # Payload 0: Packed FP4 tokens (uint8)
    packed_hidden_size = hidden_size // 2
    packed_hidden_states = torch.randint(
        0, 256, (local_num_tokens, packed_hidden_size), dtype=torch.uint8, device="cuda"
    )
    payloads.append(packed_hidden_states)

    # Payload 1: Scaling factors (fp8)
    num_elts_per_sf = 16
    num_scaling_factors = hidden_size // num_elts_per_sf
    scaling_factors = torch.randn(
        local_num_tokens, num_scaling_factors, dtype=torch.float32, device="cuda"
    )  #  .to(torch.float8_e4m3fn) TODO: Test failed.
    scaling_factors += rank
    payloads.append(scaling_factors)

    # Payload 2: token_selected_experts
    payloads.append(token_selected_experts)

    # Payload 3: token_final_scales (bfloat16)
    token_final_scales = torch.rand(
        local_num_tokens, top_k, dtype=torch.bfloat16, device="cuda"
    )

    # Construct the data to contain info about send rank and local_token_idx, which is used for debugging
    # token_final_scales[:, 0] = rank
    # token_final_scales[:, 1] = torch.linspace(0, local_num_tokens - 1, local_num_tokens, dtype=torch.bfloat16, device='cuda')

    payloads.append(token_final_scales)
    return payloads, 2


def make_bfloat16_payloads(
    local_num_tokens: int,
    hidden_size: int,
    top_k: int,
    rank: int,
    token_selected_experts: torch.Tensor,
) -> tuple[list, int]:
    """Create bfloat16 test payloads matching nvfp4 structure but without scaling factors."""
    payloads = []

    # Payload 0: Hidden states (bfloat16)
    hidden_states = torch.randn(
        local_num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    # Add rank-specific pattern for verification
    hidden_states += rank
    payloads.append(hidden_states)

    # Payload 1: token_selected_experts
    payloads.append(token_selected_experts)

    # Payload 2: token_final_scales (bfloat16) - similar to nvfp4's payload 4
    token_final_scales = torch.rand(
        local_num_tokens, top_k, dtype=torch.bfloat16, device="cuda"
    )

    # Optional: Construct the data that is easier to debug
    # token_final_scales[:, 0] = rank
    # token_final_scales[:, 1] = torch.linspace(0, local_num_tokens - 1, local_num_tokens, dtype=torch.bfloat16, device='cuda')

    payloads.append(token_final_scales)

    return payloads, 1


def run_moe_a2a_dispatch_single_rank(
    ep_size,
    all_num_tokens,
    top_k,
    num_experts_per_rank,
    hidden_size,
    invalid_token_expert_id,
):
    """Worker function for MPI testing."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # get local rank
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    node_local_rank = node_comm.Get_rank()
    torch.cuda.set_device(node_local_rank)

    check_any_rank_failed()

    mapping = Mapping(
        rank=rank,
        tp_size=ep_size,
        moe_ep_size=ep_size,
        world_size=ep_size,
        gpus_per_node=ep_size,
        pp_size=1,
        cp_size=1,
    )

    # Create MoeAlltoAll manager
    max_num_tokens = max(all_num_tokens)

    moe_a2a = MoeAlltoAll(
        mapping,
        max_num_tokens,
        top_k,
        ep_size * num_experts_per_rank,
        hidden_size=hidden_size,
    )

    # Get the number of tokens for this specific rank (same as single-GPU)
    rank_local_tokens = all_num_tokens[rank]

    # Generate data using helper functions
    token_selected_experts = generate_token_selected_experts(
        rank_local_tokens, ep_size, num_experts_per_rank, top_k
    )
    payloads, expert_id_payload_index = make_nvfp4_payloads(
        rank_local_tokens, hidden_size, top_k, rank, token_selected_experts
    )

    check_any_rank_failed()

    recv_tensors = moe_a2a.dispatch(
        token_selected_experts,
        payloads,
        max_num_tokens,
        invalid_token_expert_id=invalid_token_expert_id,
        expert_id_payload_index=expert_id_payload_index,
    )

    # Read counters and compact routing tensors from workspace
    send_counters_offset = moe_a2a.metainfo[
        MoeAlltoAll._METAINFO_INDEX["SEND_COUNTERS_OFFSET_INDEX"]
    ].item()
    recv_counters_offset = moe_a2a.metainfo[
        MoeAlltoAll._METAINFO_INDEX["RECV_COUNTERS_OFFSET_INDEX"]
    ].item()
    topk_target_ranks_offset = moe_a2a.metainfo[
        MoeAlltoAll._METAINFO_INDEX["TOPK_TARGET_RANKS_OFFSET_INDEX"]
    ].item()
    topk_send_indices_offset = moe_a2a.metainfo[
        MoeAlltoAll._METAINFO_INDEX["TOPK_SEND_INDICES_OFFSET_INDEX"]
    ].item()

    send_counters = (
        moe_a2a.workspace[
            rank, send_counters_offset : send_counters_offset + ep_size * 4
        ]
        .view(torch.int32)
        .cpu()
    )
    recv_counters = (
        moe_a2a.workspace[
            rank, recv_counters_offset : recv_counters_offset + ep_size * 4
        ]
        .view(torch.int32)
        .cpu()
    )
    topk_target_ranks = (
        moe_a2a.workspace[
            rank,
            topk_target_ranks_offset : topk_target_ranks_offset
            + max_num_tokens * top_k * 4,
        ]
        .view(torch.int32)
        .view(max_num_tokens, top_k)
        .cpu()
    )
    topk_send_indices = (
        moe_a2a.workspace[
            rank,
            topk_send_indices_offset : topk_send_indices_offset
            + max_num_tokens * top_k * 4,
        ]
        .view(torch.int32)
        .view(max_num_tokens, top_k)
        .cpu()
    )

    # Return results to be collected (move to CPU for MPI transfer)
    return (
        token_selected_experts.cpu(),
        [p.cpu() for p in payloads],
        [rt.cpu() for rt in recv_tensors],
        send_counters,
        topk_send_indices,
        topk_target_ranks,
        recv_counters,
        expert_id_payload_index,
    )


def verify_dispatch(
    all_token_selected_experts,
    all_payloads,
    all_recv_tensors,
    all_send_counters,
    all_topk_send_indices,
    all_topk_target_ranks,
    all_recv_counters,
    ep_size,
    all_num_tokens,
    top_k,
    num_experts_per_rank,
    expert_id_payload_index,
    invalid_token_expert_id,
):
    """Verify dispatch results including actual content verification"""

    max_num_tokens = max(all_num_tokens)
    # Verify dimensions and dtypes
    for send_rank in range(ep_size):
        local_num_tokens = all_num_tokens[send_rank]

        token_selected_experts = all_token_selected_experts[send_rank]
        assert len(token_selected_experts.shape) == 2, (
            "token_selected_experts should be a 2D tensor"
        )
        assert token_selected_experts.dtype == torch.int32, (
            "token_selected_experts should be a 32-bit integer tensor"
        )
        assert token_selected_experts.shape[0] == local_num_tokens, (
            "token_selected_experts.shape[0] should be local_num_tokens"
        )
        assert token_selected_experts.shape[1] == top_k, (
            "token_selected_experts.shape[1] should be top_k"
        )

        payloads = all_payloads[send_rank]
        recv_tensors = all_recv_tensors[send_rank]
        num_payloads = len(payloads)
        assert len(recv_tensors) == num_payloads, (
            "recv_tensors should have the same number of payloads as payloads"
        )
        for i in range(num_payloads):
            payload = payloads[i]
            assert len(payload.shape) == 2, "payload should be a 2D tensor"
            assert payload.shape[0] == local_num_tokens, (
                "payload.shape[0] should be local_num_tokens"
            )

            recv_tensor = recv_tensors[i]
            assert len(recv_tensor.shape) == 3, "recv_tensor should be a 3D tensor"
            assert recv_tensor.shape[0] == ep_size, (
                "recv_tensor.shape[0] should be ep_size"
            )
            assert recv_tensor.shape[1] == max_num_tokens, (
                "recv_tensor.shape[1] should be max_num_tokens"
            )
            assert recv_tensor.shape[2] == payload.shape[1], (
                "recv_tensor.shape[2] should be payload.shape[1]"
            )
            assert recv_tensor.dtype == payload.dtype, (
                "recv_tensor.dtype should be payload.dtype"
            )

        # Verify counters and compact routing tensors
        send_counters = all_send_counters[send_rank]
        assert len(send_counters.shape) == 1, "send_counters should be a 1D tensor"
        assert send_counters.shape[0] == ep_size
        assert send_counters.dtype == torch.int32

        recv_counters = all_recv_counters[send_rank]
        assert len(recv_counters.shape) == 1, "recv_counters should be a 1D tensor"
        assert recv_counters.shape[0] == ep_size
        assert recv_counters.dtype == torch.int32

        topk_send_indices = all_topk_send_indices[send_rank]
        topk_target_ranks = all_topk_target_ranks[send_rank]
        assert topk_send_indices.shape == (max_num_tokens, top_k), (
            "topk_send_indices shape"
        )
        assert topk_target_ranks.shape == (max_num_tokens, top_k), (
            "topk_target_ranks shape"
        )
        assert topk_send_indices.dtype == torch.int32
        assert topk_target_ranks.dtype == torch.int32

    # Verify send_counters per (send_rank -> target_rank)
    for send_rank in range(ep_size):
        expected_sends = {}
        token_experts = all_token_selected_experts[send_rank]
        sent_to_rank = set()

        for token_idx in range(token_experts.shape[0]):
            experts = token_experts[token_idx]
            target_ranks = compute_target_rank_id(experts, num_experts_per_rank)
            sent_to_rank.clear()

            for target_rank in target_ranks.tolist():
                if target_rank not in sent_to_rank:
                    if target_rank not in expected_sends:
                        expected_sends[target_rank] = 0
                    expected_sends[target_rank] += 1
                    sent_to_rank.add(target_rank)

        for target_rank in range(ep_size):
            expected_to_rank = expected_sends.get(target_rank, 0)
            actual_to_rank = all_send_counters[send_rank][target_rank].item()
            assert actual_to_rank == expected_to_rank, (
                f"Rank {send_rank} sent {actual_to_rank} tokens to rank {target_rank}, expected {expected_to_rank}"
            )

    # Verify recv_counters match send_counters
    for recv_rank in range(ep_size):
        for send_rank in range(ep_size):
            expected_recv = all_send_counters[send_rank][recv_rank].item()
            actual_recv = all_recv_counters[recv_rank][send_rank].item()
            assert actual_recv == expected_recv, (
                f"Rank {recv_rank} received {actual_recv} tokens from rank {send_rank}, expected {expected_recv}"
            )

    # Verify payload content using topk_send_indices and topk_target_ranks
    for send_rank in range(ep_size):
        token_selected_experts = all_token_selected_experts[send_rank]
        payloads = all_payloads[send_rank]
        topk_send_indices = all_topk_send_indices[send_rank]
        topk_target_ranks = all_topk_target_ranks[send_rank]
        local_num_tokens = all_num_tokens[send_rank]

        for token_idx in range(local_num_tokens):
            experts = token_selected_experts[token_idx]
            target_ranks = compute_target_rank_id(experts, num_experts_per_rank)
            # Deduplicate target ranks per token
            topk_target_ranks_ref = target_ranks.clone()
            seen = set()
            for kk in range(top_k):
                tr = int(topk_target_ranks_ref[kk].item())
                if tr in seen:
                    topk_target_ranks_ref[kk] = -1
                else:
                    seen.add(tr)

            assert (
                topk_target_ranks[token_idx, :].tolist()
                == topk_target_ranks_ref.tolist()
            )

            for k in range(top_k):
                dst_pos = topk_send_indices[token_idx, k].item()
                target_rank = topk_target_ranks[token_idx, k].item()
                if dst_pos == -1:
                    assert target_rank == -1
                    continue
                recv_tensors = all_recv_tensors[target_rank]
                for payload_idx, payload in enumerate(payloads):
                    recv_tensor = recv_tensors[payload_idx]
                    source_data = payload[token_idx]
                    received_data = recv_tensor[send_rank, dst_pos]
                    torch.testing.assert_close(
                        received_data, source_data, atol=0, rtol=0
                    )

    # Verify token_selected_experts of invalid tokens are correctly sanitized
    for recv_rank in range(ep_size):
        expert_ids_recv = all_recv_tensors[recv_rank][expert_id_payload_index]
        for source_rank in range(ep_size):
            valid = int(all_recv_counters[recv_rank][source_rank].item())
            for token_idx in range(max_num_tokens):
                token_expert_ids = expert_ids_recv[source_rank, token_idx]
                if token_idx >= valid:
                    assert torch.all(token_expert_ids == invalid_token_expert_id)


def moe_a2a_dispatch_test_impl(distribution, top_k):
    """Test MoE A2A dispatch operation."""
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    ep_size = world_size

    if distribution == "random":
        torch.manual_seed(0xD5)
        all_num_tokens = torch.randint(1, 100, (world_size,)).tolist()
    elif distribution == "uniform":
        all_num_tokens = [50] * world_size
    else:
        pytest.skip(f"Invalid distribution: {distribution}")

    try:
        MnnvlMemory.initialize()
        if not mnnvl_available():
            pytest.skip(
                "MNNVL not supported on this system or container lacks SYS_PTRACE capability"
            )
    except Exception:
        pytest.skip("MNNVL not supported on this system")

    hidden_size = 1024
    num_experts_per_rank = max(8, (top_k + ep_size - 1) // ep_size)
    invalid_token_expert_id = -1

    check_any_rank_failed()

    # Check all ranks have the same all_num_tokens
    gathered_all_num_tokens = comm.allgather(all_num_tokens)
    assert all(i == all_num_tokens for i in gathered_all_num_tokens[1:]), (
        "all_num_tokens should be the same"
    )

    # Run dispatch on this rank
    result = run_moe_a2a_dispatch_single_rank(
        ep_size,
        all_num_tokens,
        top_k,
        num_experts_per_rank,
        hidden_size,
        invalid_token_expert_id,
    )

    check_any_rank_failed()

    # Gather results from all ranks
    all_results = comm.allgather(result)

    # Extract results
    all_token_selected_experts = [r[0] for r in all_results]
    all_payloads = [r[1] for r in all_results]
    all_recv_tensors = [r[2] for r in all_results]
    all_send_counters = [r[3] for r in all_results]
    all_topk_send_indices = [r[4] for r in all_results]
    all_topk_target_ranks = [r[5] for r in all_results]
    all_recv_counters = [r[6] for r in all_results]
    all_expert_id_payload_index = [r[7] for r in all_results]
    expert_id_payload_index = all_expert_id_payload_index[0]

    assert all(i == expert_id_payload_index for i in all_expert_id_payload_index), (
        "all_expert_id_payload_index should be the same"
    )

    # Verify dispatch results with full counter verification
    verify_dispatch(
        all_token_selected_experts,
        all_payloads,
        all_recv_tensors,
        all_send_counters,
        all_topk_send_indices,
        all_topk_target_ranks,
        all_recv_counters,
        ep_size,
        all_num_tokens,
        top_k,
        num_experts_per_rank,
        expert_id_payload_index,
        invalid_token_expert_id,
    )


@pytest.mark.parametrize(
    "distribution,top_k",
    [
        ("random", 1),  # topk=1 with random distribution
        ("uniform", 1),  # topk=1 with uniform distribution
        ("random", 2),  # topk=2 with random distribution
        ("uniform", 2),  # topk=2 with uniform distribution
        ("random", 8),  # topk=8 with random distribution
        ("uniform", 8),  # topk=8 with uniform distribution
    ],
)
def test_moe_a2a_dispatch(distribution, top_k):
    """Test MoE A2A dispatch operation."""
    safe_run(moe_a2a_dispatch_test_impl, distribution, top_k)


def moe_a2a_dispatch_moe_combine_test_impl(distribution, top_k):
    """Test full MoE A2A dispatch + expert processing + combine cycle."""

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    ep_size = world_size

    if distribution == "random":
        torch.manual_seed(0xD5)
        all_num_tokens = torch.randint(1, 100, (world_size,)).tolist()
    elif distribution == "uniform":
        all_num_tokens = [50] * world_size
    else:
        pytest.skip(f"Invalid distribution: {distribution}")

    try:
        MnnvlMemory.initialize()
        if not mnnvl_available():
            pytest.skip(
                "MNNVL not supported on this system or container lacks SYS_PTRACE capability"
            )
    except Exception:
        pytest.skip("MNNVL not supported on this system")

    # get local rank
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    node_local_rank = node_comm.Get_rank()
    torch.cuda.set_device(node_local_rank)

    check_any_rank_failed()

    # Check all ranks have the same all_num_tokens
    gathered_all_num_tokens = comm.allgather(all_num_tokens)
    assert all(i == all_num_tokens for i in gathered_all_num_tokens), (
        "all_num_tokens should be the same"
    )

    hidden_size = 2880  # gpt-oss
    num_experts_per_rank = 8
    mapping = Mapping(
        rank=rank,
        moe_ep_size=world_size,
        tp_size=world_size,
        world_size=world_size,
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

    rank_experts = create_experts(
        num_experts_per_rank, hidden_size, rank, "cuda", dtype=torch.bfloat16
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
        hidden_size=hidden_size,
    )

    check_any_rank_failed()

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
    moe_output.copy_(
        fake_moe(
            hidden_states_recv.view(
                ep_size * max_num_tokens, hidden_states_recv.shape[-1]
            ),
            token_selected_experts_recv.view(
                ep_size * max_num_tokens, token_selected_experts_recv.shape[-1]
            ),
            token_final_scales_recv.view(
                ep_size * max_num_tokens, token_final_scales_recv.shape[-1]
            ),
            rank_experts,  # experts for current rank
            is_ep=True,
            ep_rank=rank,
            num_experts_per_rank=num_experts_per_rank,
        ).view(ep_size, max_num_tokens, hidden_size)
    )

    check_any_rank_failed()

    # Combine
    combined_output = moe_a2a.combine(
        payload=moe_output,
        runtime_max_tokens_per_rank=max_num_tokens,
        payload_in_workspace=True,
    )

    # Verify against reference
    num_matches = (
        torch.isclose(combined_output, reference_output, atol=2e-2, rtol=2e-2)
        .sum()
        .item()
    )
    match_rate = num_matches / combined_output.numel()
    match_threshold = 0.99

    # The accumulation order is not the same for the reference and the combine. For topk=8 this means that we see some accumulated errors for bf16. We tolerate up to 1% mismatches.
    assert match_rate >= match_threshold, (
        f"Sample match rate {match_rate:.2%} is below threshold "
        f"({combined_output.numel() - num_matches}/{combined_output.numel()} mismatches, expected >={match_threshold:.2%})"
    )

    # torch.testing.assert_close(combined_output, reference_output, rtol=6e-2, atol=6e-2)

    check_any_rank_failed()


@pytest.mark.parametrize(
    "distribution,top_k",
    [
        ("random", 1),  # topk=1 with random distribution
        ("uniform", 1),  # topk=1 with uniform distribution
        ("random", 2),  # topk=2 with random distribution
        ("uniform", 2),  # topk=2 with uniform distribution
        ("random", 8),  # topk=8 with random distribution
        ("uniform", 8),  # topk=8 with uniform distribution
    ],
)
def test_moe_a2a_dispatch_moe_combine(distribution, top_k):
    """Test full MoE A2A dispatch + expert processing + combine cycle."""
    safe_run(moe_a2a_dispatch_moe_combine_test_impl, distribution, top_k)


if __name__ == "__main__":
    # Run with: mpirun -n 2 python -m pytest tests/comm/test_mnnvl_a2a.py -v
    pytest.main([__file__, "-v", "-s"])
