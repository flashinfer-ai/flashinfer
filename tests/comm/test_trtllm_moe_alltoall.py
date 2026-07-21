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

from enum import IntEnum
import itertools
import pytest
import pynvml
import torch

from flashinfer.comm.mapping import Mapping
from flashinfer.fused_moe.utils import make_random_topk_ids
from flashinfer.tllm_enums import SfLayout
from flashinfer.utils import get_compute_capability
from flashinfer import mxfp4_quantize, nvfp4_quantize
from flashinfer.fp4_quantization import e2m1_and_ufp8sf_scale_to_float
from tests.utils_fp8 import mxfp8_quantize_reference

import flashinfer.comm.trtllm_moe_alltoall as trtllm_moe_alltoall

from .conftest import mnnvl_available

pynvml.nvmlInit()


@pytest.fixture(autouse=True, scope="session")
def setup_test_environment():
    """Set up torch seed for deterministic tests."""
    torch.manual_seed(0xD5)
    yield


NUM_LORA_ADAPTERS = 8

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

LARGER_PAYLOADS_PARAMS = [
    (8, 16, 2048, 5),  # Medium input, 8 ranks, 5 payloads
    (8, 16, 2048, 6),  # Medium input, 8 ranks, 6 payloads
]


class CombineQuantMode(IntEnum):
    NONE = 0
    MXFP8 = 1
    MXFP4 = 2
    NVFP4 = 3


# (world_size, num_tokens, vector_dim, top_k, dtype, payload_in_workspace)
COMBINE_PARAMS = [
    # Coverage for popular model specifications
    (4, 16, 4096, 2, torch.bfloat16, True),  # Mixtral-8x7B
    (4, 16, 2880, 4, torch.bfloat16, True),  # GPT-OSS-120B
    (8, 16, 5120, 6, torch.bfloat16, True),  # DeepSeek-V2
    (8, 16, 7168, 8, torch.bfloat16, True),  # DeepSeek-V3
    (8, 16, 4096, 8, torch.bfloat16, True),  # Qwen3-235B-A22B
    (8, 16, 4096, 10, torch.bfloat16, True),  # Qwen3.5-397B-A17B
    (8, 16, 4096, 16, torch.bfloat16, True),  # Fake, no known model with top_k=16
    (8, 16, 4096, 22, torch.bfloat16, True),  # Nemotron-3-Super-120B-A12B
]


# (quant_mode, sf_layout)
QUANT_PARAMS = [
    (CombineQuantMode.NONE, SfLayout.layout_linear),
    (CombineQuantMode.MXFP8, SfLayout.layout_linear),
    (CombineQuantMode.MXFP8, SfLayout.layout_128x4),
    (CombineQuantMode.MXFP8, SfLayout.layout_8x4),
    (CombineQuantMode.MXFP4, SfLayout.layout_128x4),
    (CombineQuantMode.NVFP4, SfLayout.layout_128x4),
]


LORA_COMBINE_PARAMS = [
    # (world_size, num_tokens, vector_dim, top_k, dtype, payload_in_workspace, quant_mode, sf_layout, use_lora)
    (
        8,
        16,
        7168,
        8,
        torch.bfloat16,
        True,
        CombineQuantMode.NONE,
        SfLayout.layout_linear,
        True,
    ),  # DeepSeek-V3 with LoRA
    (
        4,
        16,
        4096,
        2,
        torch.bfloat16,
        True,
        CombineQuantMode.NONE,
        SfLayout.layout_linear,
        True,
    ),  # Mixtral-8x7B with LoRA
    (
        8,
        16,
        4096,
        8,
        torch.bfloat16,
        False,
        CombineQuantMode.NONE,
        SfLayout.layout_linear,
        True,
    ),  # LoRA + payload not in workspace
]


def ceil_div(a, b):
    return (a + b - 1) // b


def _compute_sf_size(
    combine_quant_mode: CombineQuantMode,
    num_rows: int,
    hidden_size: int,
    sf_layout: SfLayout,
):
    """Number of scale-factor bytes needed for an [num_rows, hidden_size] tensor."""
    assert combine_quant_mode != CombineQuantMode.NONE
    sf_vec_size = 16 if combine_quant_mode == CombineQuantMode.NVFP4 else 32
    cols = ceil_div(hidden_size, sf_vec_size)

    if sf_layout == SfLayout.layout_linear:
        return num_rows * cols
    elif sf_layout == SfLayout.layout_128x4:
        padded_row = (num_rows + 127) // 128 * 128
        padded_col = (cols + 3) // 4 * 4
        return padded_row * padded_col
    elif sf_layout == SfLayout.layout_8x4:
        padded_row = (num_rows + 7) // 8 * 8
        padded_col = (cols + 3) // 4 * 4
        return padded_row * padded_col
    else:
        raise ValueError(f"Unsupported sf_layout: {sf_layout}")


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


def test_moe_a2a_combine_rejects_cpu_output_before_jit(monkeypatch):
    monkeypatch.setattr(
        trtllm_moe_alltoall,
        "get_moe_alltoall_module",
        lambda: pytest.fail("invalid output should be rejected before JIT loading"),
    )
    payload = torch.empty((1, 1, 1))

    with pytest.raises(ValueError, match="output must be a CUDA tensor"):
        trtllm_moe_alltoall.moe_a2a_combine(
            payload,
            1,
            torch.empty(0),
            torch.empty(0),
            1,
            0,
            1,
            1,
            0,
            output=torch.empty((1, 1)),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_moe_a2a_combine_rejects_noncontiguous_output_before_jit(monkeypatch):
    monkeypatch.setattr(
        trtllm_moe_alltoall,
        "get_moe_alltoall_module",
        lambda: pytest.fail("invalid output should be rejected before JIT loading"),
    )
    payload = torch.empty((1, 1, 2), device="cuda")
    output = torch.empty((2, 2), device="cuda").T

    with pytest.raises(ValueError, match="output must be contiguous"):
        trtllm_moe_alltoall.moe_a2a_combine(
            payload,
            1,
            torch.empty(0, device="cuda"),
            torch.empty(0),
            1,
            0,
            1,
            1,
            0,
            output=output,
        )


@pytest.mark.parametrize(
    "num_tokens,vector_dim,num_experts,top_k",
    SINGLE_GPU_PARAMS,
)
@pytest.mark.parametrize("payload_in_workspace", [False, True])
@pytest.mark.parametrize("use_output_buffer", [False, True])
@pytest.mark.skipif(
    not mnnvl_available(),
    reason="Mnnvl memory is not supported on this platform or container lacks SYS_PTRACE capability",
)
def test_moe_alltoall_single_gpu(
    num_tokens,
    vector_dim,
    num_experts,
    top_k,
    payload_in_workspace,
    use_output_buffer,
):
    """Test MOE alltoall communication on single GPU."""
    torch.cuda.set_device(0)
    # Create a random input tensor
    dtypes = [torch.bfloat16, torch.float16, torch.int32, torch.uint8, torch.int32]
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

    if payload_in_workspace:
        combine_tensor = moe_a2a.get_combine_payload_tensor_in_workspace(
            num_tokens,
            input_tensors[hidden_state_index].shape[-1],
            input_tensors[hidden_state_index].dtype,
        )
        combine_tensor.copy_(output_tensors[hidden_state_index])
    else:
        combine_tensor = output_tensors[hidden_state_index].clone()

    output_buffer = (
        torch.empty_like(input_tensors[hidden_state_index])
        if use_output_buffer
        else None
    )
    output = moe_a2a.combine(
        combine_tensor,
        num_tokens,
        payload_in_workspace=payload_in_workspace,
        output=output_buffer,
    )

    if output_buffer is not None:
        assert output is output_buffer
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
    output_dtype=None,
    output_scales_list=None,
    output_scalar_scale=1.0,
    sf_layout=SfLayout.layout_linear,
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
                    output_dtype=output_dtype,
                    output_scales=(
                        output_scales_list[rank]
                        if output_scales_list is not None
                        else None
                    ),
                    output_scalar_scale=output_scalar_scale,
                    sf_layout=sf_layout,
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

    # 5 payloads (max): the trailing int32 represents the LoRA adapter ID slot.
    dtypes = [torch.float16, torch.bfloat16, torch.int32, torch.uint8, torch.int32]
    # Create a random input tensor
    input_tensors = [
        make_payload(num_tokens * world_size, vector_dim * (i + 1), dtype)
        for i, dtype in enumerate(dtypes)
    ]

    token_selected_experts = make_random_topk_ids(
        num_experts=world_size,
        num_tokens=world_size * num_tokens,
        top_k=1,
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


@pytest.mark.parametrize("top_k", [6, 8, 10, 16, 22])
def test_moe_alltoall_compact_ep4_dispatch(top_k):
    """Test the compact EP4 dispatch path with repeated destination ranks."""
    torch.cuda.set_device(0)
    world_size = 4
    num_tokens = 4
    num_experts = 64
    total_tokens = world_size * num_tokens
    num_experts_per_rank = num_experts // world_size

    input_tensors = [
        make_payload(total_tokens, 256, torch.int32),  # 1024 bytes per token
        make_payload(total_tokens, 128, torch.uint8),
        make_payload(total_tokens, 22, torch.int32),
        make_payload(total_tokens, 22, torch.int32),
    ]

    # Rotate through experts so each token has repeated destinations and the set
    # of destination ranks varies with both the token and top-k width.
    expert_ids = torch.arange(
        total_tokens * top_k, dtype=torch.int32, device=torch.device("cuda")
    ).reshape(total_tokens, top_k)
    token_selected_experts = (
        expert_ids * 7
        + torch.arange(
            total_tokens, dtype=torch.int32, device=torch.device("cuda")
        ).unsqueeze(1)
    ).remainder(num_experts)

    output_tensors, _, _, _ = dispatch_from_single_rank(
        input_tensors,
        token_selected_experts,
        world_size,
        num_experts,
        num_tokens,
    )
    target_ranks = token_selected_experts.div(
        num_experts_per_rank, rounding_mode="floor"
    )

    for rank in range(world_size):
        expected_token_mask = (target_ranks == rank).any(dim=1)
        for actual, ref in zip(output_tensors[rank], input_tensors, strict=True):
            actual = actual.flatten(end_dim=1)
            actual = actual[actual.any(dim=1)]
            expected = ref[expected_token_mask]
            actual, _ = torch.sort(actual, dim=0)
            expected, _ = torch.sort(expected, dim=0)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "world_size,num_tokens,vector_dim,num_payloads", LARGER_PAYLOADS_PARAMS
)
def test_moe_alltoall_dispatch_larger_payloads_single_gpu(
    world_size,
    num_tokens,
    vector_dim,
    num_payloads,
):
    """Test dispatch with the maximum supported payload count."""
    torch.cuda.set_device(0)
    check_sufficient_sm_count(num_tokens, world_size)

    total_tokens = num_tokens * world_size
    input_tensors = [
        (
            torch.arange(
                1,
                total_tokens * vector_dim + 1,
                dtype=torch.int32,
                device=torch.device("cuda"),
            ).reshape(total_tokens, vector_dim)
            + payload_idx * 1000
        )
        for payload_idx in range(num_payloads)
    ]

    token_selected_experts = (
        torch.arange(total_tokens, dtype=torch.int32, device=torch.device("cuda"))
        .remainder(world_size)
        .reshape(total_tokens, 1)
        .contiguous()
    )

    output_tensors, _, _, _ = dispatch_from_single_rank(
        input_tensors, token_selected_experts, world_size, world_size, num_tokens
    )

    for rank in range(world_size):
        assert len(output_tensors[rank]) == num_payloads
        token_selected_experts_indices = (
            token_selected_experts.flatten() == rank
        ).nonzero(as_tuple=False)

        for actual, ref in zip(output_tensors[rank], input_tensors, strict=True):
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
    token_selected_experts = make_random_topk_ids(
        num_experts=world_size,
        num_tokens=world_size * num_tokens,
        top_k=1,
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
    lora_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Apply a deterministic fake MoE transformation for validation.

    Each expert applies a predictable scale: (expert_id + 1.0) / num_experts + 0.5
    When lora_ids is provided, the LoRA adapter ID adds an integer step: scale += lora_id + 1.0

    Args:
        hidden_states: Input tensor [num_tokens, hidden_size] or [world_size, num_tokens, hidden_size]
        token_selected_experts: Expert assignments [num_tokens, top_k] or [world_size, num_tokens, top_k]
        num_experts: Total number of experts
        is_ep: If True, only process experts assigned to this rank
        ep_rank: Rank for expert parallel filtering
        num_experts_per_rank: Number of experts per rank
        lora_ids: Per-token LoRA adapter IDs [num_tokens] int32, or None

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
            if lora_ids is not None:
                scale += lora_ids[token_idx].item() + 1.0
            results.append(hidden_states[token_idx] * scale)

        # Summing the results after is closer to the actual implementation as we do a tree reduction.
        if results:
            processed_states[token_idx] = torch.sum(
                torch.stack(results, dim=0), dim=0, dtype=torch.float32
            ).to(processed_states.dtype)
    return processed_states.view(target_shape)


@pytest.mark.parametrize(
    "world_size,num_tokens,vector_dim,top_k,dtype,payload_in_workspace,quant_mode,sf_layout,use_lora",
    [(*x, *y, False) for x, y in itertools.product(COMBINE_PARAMS, QUANT_PARAMS)]
    + LORA_COMBINE_PARAMS,
)
def test_moe_combine_multi_rank_single_gpu(
    world_size,
    num_tokens,
    vector_dim,
    top_k,
    dtype,
    payload_in_workspace,
    quant_mode,
    sf_layout,
    use_lora,
):
    torch.cuda.set_device(0)
    compute_capability = get_compute_capability(torch.device("cuda"))
    compute_capability_number = compute_capability[0] * 10 + compute_capability[1]
    if quant_mode != CombineQuantMode.NONE and compute_capability_number < 100:
        pytest.skip(
            f"Combine-quantization fusion requires CUDA platform >= 100, got {compute_capability_number}."
        )

    check_sufficient_sm_count(num_tokens, world_size)
    max_world_size = 16
    assert world_size <= max_world_size, (
        f"should run with world_size at most {max_world_size}"
    )

    num_experts = world_size * top_k
    total_tokens = num_tokens * world_size

    token_selected_experts_index = 0
    hidden_state_index = 1

    token_selected_experts = torch.empty(
        total_tokens, top_k, dtype=torch.int32, device=torch.device("cuda")
    )

    for i in range(total_tokens):
        # Include one extra expert to represent invalid expert IDs
        token_selected_experts[i] = torch.randperm(
            num_experts, dtype=torch.int32, device=torch.device("cuda")
        )[:top_k]
    token_selected_experts = token_selected_experts.contiguous()

    # Create a random input tensor
    reference_tensor = make_payload(total_tokens, vector_dim, dtype)
    input_tensors = [
        token_selected_experts,
        reference_tensor,
        make_payload(
            total_tokens, 1, torch.uint8
        ),  # Some extra payload to test combine alignment logic
    ]

    lora_ids = None
    lora_id_payload_index = None
    if use_lora:
        lora_ids = torch.randint(
            0,
            NUM_LORA_ADAPTERS,
            (total_tokens,),
            dtype=torch.int32,
            device=torch.device("cuda"),
        )
        lora_id_payload_index = len(input_tensors)
        # Dispatch kernel requires 2D payloads [num_tokens, elements_per_token]
        input_tensors.append(lora_ids.unsqueeze(-1))

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
        recv_lora_ids = None
        if use_lora:
            recv_lora_ids = output_tensors[rank][lora_id_payload_index].flatten()
        inplace_combine_tensors[rank].copy_(
            fake_moe(
                output_tensors[rank][hidden_state_index],
                output_tensors[rank][token_selected_experts_index],
                num_experts,
                is_ep=True,
                ep_rank=rank,
                num_experts_per_rank=num_experts // world_size,
                lora_ids=recv_lora_ids,
            )
        )

    output_scalar_scale = 1.0
    if quant_mode != CombineQuantMode.NONE:
        output_scalar_scale = (
            2.5  # arbitrary non-one scalar to test scaling path in the kernel
        )
        # High-precision (bf16) combine output. The quantized kernel output is
        # validated by dequantizing it and comparing against this reference.
        reference_result = combine_from_single_rank(
            inplace_combine_tensors,
            num_tokens,
            top_k,
            all_workspaces,
            metainfo,
            world_size,
            combine_payload_offsets,
            payload_in_workspace=payload_in_workspace,
        )
        sf_size = _compute_sf_size(quant_mode, num_tokens, vector_dim, sf_layout)
        if quant_mode == CombineQuantMode.MXFP8:
            output_dtype = torch.float8_e4m3fn
            sf_dtype = torch.uint8  # UE8M0 scale factors, packed as bytes
        else:
            # MXFP4 / NVFP4: two FP4 (e2m1) values packed per uint8 byte. The scale
            # dtype is how the binding distinguishes the two: uint8 (UE8M0) -> MXFP4,
            # float8_e4m3fn (UE4M3) -> NVFP4.
            output_dtype = torch.uint8
            sf_dtype = (
                torch.uint8
                if quant_mode == CombineQuantMode.MXFP4
                else torch.float8_e4m3fn
            )
        output_scales_list = [
            torch.zeros(sf_size, dtype=sf_dtype, device=torch.device("cuda"))
            for _ in range(world_size)
        ]
    else:
        output_dtype = None
        output_scales_list = None

    combine_results = combine_from_single_rank(
        inplace_combine_tensors,
        num_tokens,
        top_k,
        all_workspaces,
        metainfo,
        world_size,
        combine_payload_offsets,
        payload_in_workspace=payload_in_workspace,
        output_dtype=output_dtype,
        output_scales_list=output_scales_list,
        sf_layout=sf_layout,
        output_scalar_scale=output_scalar_scale,
    )

    if quant_mode == CombineQuantMode.NONE:
        reference_result = fake_moe(
            input_tensors[hidden_state_index],
            token_selected_experts,
            num_experts,
            lora_ids=lora_ids,
        )
        for rank in range(world_size):
            torch.testing.assert_close(
                combine_results[rank],
                reference_result[rank * num_tokens : (rank + 1) * num_tokens],
                atol=1.5e-2,
                rtol=1.5e-2,
            )
    elif quant_mode == CombineQuantMode.MXFP8:
        for rank in range(world_size):
            ref_fp8, ref_sf = mxfp8_quantize_reference(
                reference_result[rank],
                sf_swizzle_layout=sf_layout,
            )
            # Compare FP8 values via float32 cast (assert_close doesn't accept fp8 dtype).
            torch.testing.assert_close(
                combine_results[rank].to(torch.float32),
                ref_fp8.to(torch.float32),
                atol=1.5e-2,
                rtol=1.5e-2,
            )
            torch.testing.assert_close(
                output_scales_list[rank],
                ref_sf,
                atol=0,
                rtol=0,
            )
    elif quant_mode in (CombineQuantMode.MXFP4, CombineQuantMode.NVFP4):
        is_nvfp4 = quant_mode == CombineQuantMode.NVFP4
        sf_vec_size = 16 if is_nvfp4 else 32
        # e2m1_and_ufp8sf_scale_to_float: ufp8_type 1 = UE4M3 (NVFP4), 0 = UE8M0 (MXFP4).
        ufp8_type = 1 if is_nvfp4 else 0
        is_swizzled = sf_layout != SfLayout.layout_linear
        global_sf = torch.tensor(
            [output_scalar_scale], dtype=torch.float32, device="cuda"
        )
        for rank in range(world_size):
            if is_nvfp4:
                ref_fp4, ref_sf = nvfp4_quantize(
                    reference_result[rank],
                    global_sf,
                    sfLayout=sf_layout,
                    sf_vec_size=16,
                    do_shuffle=False,
                )
            else:
                ref_fp4, ref_sf = mxfp4_quantize(reference_result[rank])
            # Packed FP4 bytes can't be compared directly (two e2m1 codes share a byte),
            # so dequantize both the kernel output and the reference identically and
            # compare in float space. Both paths share the cvt_warp_fp16_to_fp4 routine,
            # so the dequantized values should match within FP4 granularity.
            actual = e2m1_and_ufp8sf_scale_to_float(
                combine_results[rank].view(torch.uint8),
                output_scales_list[rank].view(torch.uint8).reshape(-1),
                global_sf,
                sf_vec_size,
                ufp8_type,
                is_swizzled,
            )
            expected = e2m1_and_ufp8sf_scale_to_float(
                ref_fp4.view(torch.uint8),
                ref_sf.view(torch.uint8).reshape(-1),
                global_sf,
                sf_vec_size,
                ufp8_type,
                is_swizzled,
            )
            torch.testing.assert_close(actual, expected, atol=1.5e-2, rtol=1.5e-2)
    else:
        raise ValueError(f"Unsupported quant_mode: {quant_mode}")


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
