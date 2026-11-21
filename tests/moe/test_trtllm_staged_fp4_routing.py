# Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: Apache-2.0

"""Return contracts of the staged FP4 routing entrypoints."""

import pytest
import torch

from flashinfer.fused_moe.core import get_trtllm_moe_sm100_module
from flashinfer.tllm_enums import ActivationType, RoutingMethodType, WeightLayout
from flashinfer.utils import get_compute_capability


@pytest.mark.parametrize(
    "format_name,allocate_gemm2_output,expected_length",
    [
        pytest.param("nvfp4", True, 11, id="nvfp4-unfused"),
        pytest.param("nvfp4", False, 11, id="nvfp4-fused"),
        pytest.param("mxfp4_mxfp8", True, 10, id="mxfp4-mxfp8"),
        pytest.param("mxfp4_bf16", True, 9, id="mxfp4-bf16"),
    ],
)
def test_staged_fp4_routing_outputs(
    format_name, allocate_gemm2_output, expected_length
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    if get_compute_capability(device)[0] != 10:
        pytest.skip("Staged FP4 MoE requires an SM100-family GPU")

    num_tokens, hidden_size, intermediate_size, num_experts, top_k = 17, 1024, 512, 8, 2
    is_nvfp4 = format_name == "nvfp4"
    is_bf16 = format_name == "mxfp4_bf16"
    dtype = (
        torch.uint8 if is_nvfp4 else torch.bfloat16 if is_bf16 else torch.float8_e4m3fn
    )
    hidden_states = torch.zeros(
        (num_tokens, hidden_size // 2 if is_nvfp4 else hidden_size),
        dtype=dtype,
        device=device,
    )
    expert_indices = (
        torch.arange(num_tokens * top_k, dtype=torch.int32, device=device)
        .reshape(num_tokens, top_k)
        .remainder(num_experts)
    )
    expert_weights = torch.full(
        (num_tokens, top_k), 1 / top_k, dtype=torch.bfloat16, device=device
    )
    args = [None, None, expert_indices, expert_weights, hidden_states]
    if not is_bf16:
        args.append(
            torch.ones(
                (128, hidden_size // (16 if is_nvfp4 else 32)),
                dtype=torch.float8_e4m3fn,
                device=device,
            )
        )
    # Routing-only callers may omit weights; GEMM buffers are still returned.
    args.extend([None, None, None, None])
    scale = torch.ones(num_experts, dtype=torch.float32, device=device)
    args.extend(
        [
            scale,
            scale,
            scale,
            num_experts,
            top_k,
            None,
            None,
            intermediate_size,
            0,
            num_experts,
            None,
            int(RoutingMethodType.Renormalize),
            False,
            [128, 0],
            int(WeightLayout.MajorK),
            int(ActivationType.Swiglu),
            True,
            None,
        ]
    )
    if is_nvfp4:
        args.append(allocate_gemm2_output)
    moe_op = get_trtllm_moe_sm100_module().moe_op
    outputs = getattr(moe_op, f"trtllm_moe_run_routing_fp4_{format_name}")(*args)
    assert len(outputs) == expected_length
    outputs = [torch.from_dlpack(t) if t is not None else None for t in outputs]

    expanded_to_permuted = outputs[1].long()
    assert expanded_to_permuted.numel() == num_tokens * top_k
    assert expanded_to_permuted.unique().numel() == num_tokens * top_k
    assert ((expanded_to_permuted >= 0) & (expanded_to_permuted < outputs[6])).all()
    routed_tokens = outputs[2][expanded_to_permuted].long()
    torch.testing.assert_close(
        torch.bincount(routed_tokens, minlength=num_tokens),
        torch.full((num_tokens,), top_k, dtype=torch.int64, device=device),
    )
    assert outputs[7].dtype == (torch.bfloat16 if is_bf16 else torch.uint8)
    gemm2_output = outputs[8 if is_bf16 else 9]
    assert gemm2_output.dtype == torch.bfloat16
    assert gemm2_output.shape[1] == hidden_size
    assert (gemm2_output.numel() > 0) == allocate_gemm2_output
    if is_nvfp4:
        if allocate_gemm2_output:
            assert outputs[10] is None
        else:
            assert outputs[10].dtype == torch.int32
            torch.testing.assert_close(
                outputs[10][expanded_to_permuted],
                torch.arange(num_tokens * top_k, dtype=torch.int32, device=device),
            )
