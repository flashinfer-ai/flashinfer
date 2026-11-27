"""Unit tests validating hidden_size alignment requirements in trtllm_fp4_block_scale_moe."""

import pytest
import torch

from flashinfer.fused_moe.core import (
    _validate_trtllm_fp4_hidden_size,
    trtllm_fp4_block_scale_moe,
)


def test_validate_trtllm_fp4_hidden_size():
    """Verify _validate_trtllm_fp4_hidden_size rejects NVFP4 hidden_size not divisible by 256."""
    num_tokens = 4

    # 1. hidden_size = 128 (uint8 packed fp4 has shape 64, scale shape 8) -> reject
    hs_128 = torch.zeros((num_tokens, 64), dtype=torch.uint8, device="meta")
    scale_128 = torch.zeros((num_tokens, 8), dtype=torch.float8_e4m3fn, device="meta")
    with pytest.raises(ValueError, match="multiple of 256"):
        _validate_trtllm_fp4_hidden_size(hs_128, scale_128)

    # 2. hidden_size = 384 (uint8 packed fp4 has shape 192, scale shape 24) -> reject
    hs_384 = torch.zeros((num_tokens, 192), dtype=torch.uint8, device="meta")
    scale_384 = torch.zeros((num_tokens, 24), dtype=torch.float8_e4m3fn, device="meta")
    with pytest.raises(ValueError, match="multiple of 256"):
        _validate_trtllm_fp4_hidden_size(hs_384, scale_384)

    # 3. hidden_size = 256 (uint8 packed fp4 has shape 128, scale shape 16) -> valid
    hs_256 = torch.zeros((num_tokens, 128), dtype=torch.uint8, device="meta")
    scale_256 = torch.zeros((num_tokens, 16), dtype=torch.float8_e4m3fn, device="meta")
    _validate_trtllm_fp4_hidden_size(hs_256, scale_256)

    # 4. hidden_size = 512 -> valid
    hs_512 = torch.zeros((num_tokens, 256), dtype=torch.uint8, device="meta")
    scale_512 = torch.zeros((num_tokens, 32), dtype=torch.float8_e4m3fn, device="meta")
    _validate_trtllm_fp4_hidden_size(hs_512, scale_512)

    # 5. MXFP4 (scale is per 32 elements: scale.numel() == x_numel // 32) -> allowed
    mxfp4_hs_128 = torch.zeros((num_tokens, 64), dtype=torch.uint8, device="meta")
    mxfp4_scale_128 = torch.zeros((num_tokens, 4), dtype=torch.float8_e4m3fn, device="meta")
    _validate_trtllm_fp4_hidden_size(mxfp4_hs_128, mxfp4_scale_128)


def test_trtllm_fp4_block_scale_moe_api_rejects_unaligned_nvfp4():
    """Verify trtllm_fp4_block_scale_moe entry point rejects NVFP4 hidden_size % 256 != 0."""
    num_tokens = 2
    num_experts = 4
    top_k = 2
    intermediate_size = 512
    hidden_size = 128

    routing_logits = torch.zeros((num_tokens, num_experts), dtype=torch.bfloat16, device="meta")
    hidden_states = torch.zeros((num_tokens, hidden_size // 2), dtype=torch.uint8, device="meta")
    scale = torch.zeros((num_tokens, hidden_size // 16), dtype=torch.float8_e4m3fn, device="meta")
    w1 = torch.zeros((num_experts, 2 * intermediate_size, hidden_size // 2), dtype=torch.uint8, device="meta")
    w1_s = torch.zeros((num_experts, 2 * intermediate_size, hidden_size // 16), dtype=torch.float8_e4m3fn, device="meta")
    w2 = torch.zeros((num_experts, hidden_size, intermediate_size // 2), dtype=torch.uint8, device="meta")
    w2_s = torch.zeros((num_experts, hidden_size, intermediate_size // 16), dtype=torch.float8_e4m3fn, device="meta")

    with pytest.raises(ValueError, match="multiple of 256"):
        trtllm_fp4_block_scale_moe(
            routing_logits,
            None,
            hidden_states,
            scale,
            w1,
            w1_s,
            None,
            None,
            None,
            None,
            w2,
            w2_s,
            None,
            None,
            None,
            None,
            num_experts=num_experts,
            top_k=top_k,
            n_group=None,
            topk_group=None,
            intermediate_size=intermediate_size,
            local_expert_offset=0,
            local_num_experts=num_experts,
            routed_scaling_factor=None,
        )
