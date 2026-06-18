"""
Copyright (c) 2025 by FlashInfer team.

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

import os

os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

import pytest
import torch


# ============================================================
# PyTorch Reference Implementation
# ============================================================


def reference_moe_bgmv_shrink(
    x: torch.Tensor,  # [num_tokens, hidden_dim]
    lora_a_weights: list,  # list of [max_loras, num_experts, rank, hidden_dim]
    sorted_token_ids: torch.Tensor,  # [num_pairs]
    expert_ids: torch.Tensor,  # [num_pairs]
    lora_indices: torch.Tensor,  # [num_tokens]
) -> torch.Tensor:
    """
    Reference shrink: for each (token, expert) pair, compute x @ lora_a^T.

    Returns: [num_slices, num_pairs, rank]
    """
    num_slices = len(lora_a_weights)
    num_pairs = sorted_token_ids.size(0)
    rank = lora_a_weights[0].size(2)
    device = x.device
    dtype = x.dtype

    y = torch.zeros(num_slices, num_pairs, rank, dtype=dtype, device=device)

    for pair_idx in range(num_pairs):
        token_idx = sorted_token_ids[pair_idx].item()
        if token_idx < 0 or token_idx >= x.size(0):
            continue
        lora_id = lora_indices[token_idx].item()
        if lora_id < 0:
            continue
        expert_id = expert_ids[pair_idx].item()
        x_tok = x[token_idx]  # [hidden_dim]

        for s in range(num_slices):
            # lora_a shape: [max_loras, num_experts, rank, hidden_dim]
            w_a = lora_a_weights[s][lora_id, expert_id]  # [rank, hidden_dim]
            # y[s, pair, :] = x_tok @ w_a^T
            y[s, pair_idx] = x_tok @ w_a.t()

    return y


def reference_moe_bgmv_expand(
    shrink_out: torch.Tensor,  # [num_slices, num_pairs, rank]
    lora_b_weights: list,  # list of [max_loras, num_experts, feat_out, rank]
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    lora_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """
    Reference expand: for each (token, expert) pair, compute shrink_out @ lora_b^T * topk_weight.

    Returns: [num_tokens, total_feat_out]
    """
    num_slices = len(lora_b_weights)
    num_pairs = sorted_token_ids.size(0)
    device = shrink_out.device

    feat_out_per_slice = [lora_b_weights[s].size(2) for s in range(num_slices)]
    total_feat_out = sum(feat_out_per_slice)

    y = torch.zeros(num_tokens, total_feat_out, dtype=torch.float32, device=device)

    for pair_idx in range(num_pairs):
        token_idx = sorted_token_ids[pair_idx].item()
        if token_idx < 0 or token_idx >= num_tokens:
            continue
        lora_id = lora_indices[token_idx].item()
        if lora_id < 0:
            continue
        expert_id = expert_ids[pair_idx].item()
        topk_w = topk_weights[pair_idx].item()

        col_offset = 0
        for s in range(num_slices):
            # lora_b shape: [max_loras, num_experts, feat_out, rank]
            w_b = lora_b_weights[s][lora_id, expert_id]  # [feat_out, rank]
            x_s = shrink_out[s, pair_idx]  # [rank]
            # y[token, col_offset:col_offset+feat_out] += topk_w * (x_s @ w_b^T)
            y[token_idx, col_offset : col_offset + feat_out_per_slice[s]] += topk_w * (
                x_s @ w_b.t()
            )
            col_offset += feat_out_per_slice[s]

    return y


def reference_bgmv_moe(
    x: torch.Tensor,
    lora_a_weights: list,
    lora_b_weights: list,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    lora_indices: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Full reference: shrink + expand."""
    shrink_out = reference_moe_bgmv_shrink(
        x, lora_a_weights, sorted_token_ids, expert_ids, lora_indices
    )
    y = reference_moe_bgmv_expand(
        shrink_out,
        lora_b_weights,
        sorted_token_ids,
        expert_ids,
        lora_indices,
        topk_weights,
        x.size(0),
    )
    return y


# ============================================================
# Test Fixtures
# ============================================================


def generate_test_data(
    num_tokens: int,
    hidden_size: int,
    rank: int,
    num_experts: int,
    top_k: int,
    num_loras: int,
    num_slices: int,
    dtype: torch.dtype,
    device: str = "cuda",
):
    """Generate random test data for BGMV MoE kernels."""
    # Input activations
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device) * 0.1

    # LoRA weights: [max_loras, num_experts, rank, hidden_dim] for A
    #               [max_loras, num_experts, feat_out, rank] for B
    max_loras = num_loras
    feat_out = hidden_size  # For simplicity, same as hidden_size

    lora_a_weights = [
        torch.randn(
            max_loras, num_experts, rank, hidden_size, dtype=dtype, device=device
        )
        * 0.01
        for _ in range(num_slices)
    ]
    lora_b_weights = [
        torch.randn(max_loras, num_experts, feat_out, rank, dtype=dtype, device=device)
        * 0.01
        for _ in range(num_slices)
    ]

    # Routing: each token is routed to top_k experts
    # sorted_token_ids: flattened token indices (repeated for each expert)
    # expert_ids: which expert each pair goes to
    num_pairs = num_tokens * top_k

    # Simple routing: token i goes to experts [i*top_k % num_experts, ...]
    sorted_token_ids = torch.arange(
        num_tokens, device=device, dtype=torch.int64
    ).repeat_interleave(top_k)
    expert_ids = torch.randint(
        0, num_experts, (num_pairs,), device=device, dtype=torch.int64
    )
    topk_weights = (
        torch.softmax(torch.randn(num_tokens, top_k, device=device), dim=-1)
        .view(-1)
        .to(torch.float32)
    )

    # LoRA indices: assign each token a random LoRA adapter (some may be -1 = no LoRA)
    lora_indices = torch.randint(
        -1, max_loras, (num_tokens,), device=device, dtype=torch.int64
    )
    # Ensure at least some tokens have valid LoRA
    lora_indices[: num_tokens // 2] = torch.randint(
        0, max_loras, (num_tokens // 2,), device=device, dtype=torch.int64
    )

    return {
        "x": x,
        "lora_a_weights": lora_a_weights,
        "lora_b_weights": lora_b_weights,
        "sorted_token_ids": sorted_token_ids,
        "expert_ids": expert_ids,
        "topk_weights": topk_weights,
        "lora_indices": lora_indices,
        "num_experts": num_experts,
        "num_pairs": num_pairs,
        "rank": rank,
        "num_slices": num_slices,
        "feat_out": hidden_size,
    }


# ============================================================
# Tests
# ============================================================

# BGMV MoE kernels are tested on SM80-SM90 (A100, H100, B200).
# Skip on consumer Blackwell GPUs (SM120, e.g., RTX 5090, RTX Pro 6000)
# where extended shared memory behavior may differ.
_SUPPORTED_SM = {90, 100, 103}


def _skip_if_unsupported_sm():
    """Skip test if current GPU SM version is not in the supported set."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    capability = torch.cuda.get_device_capability()
    sm = capability[0] * 10 + capability[1]
    if sm not in _SUPPORTED_SM:
        pytest.skip(
            f"BGMV MoE kernel not validated on SM{sm} "
            f"(device: {torch.cuda.get_device_name()}). "
            f"Supported: {sorted(_SUPPORTED_SM)}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestBgmvMoeShrink:
    """Test the shrink kernel against reference."""

    def setup_method(self):
        _skip_if_unsupported_sm()

    @pytest.mark.parametrize("num_tokens", [1, 4, 32])
    @pytest.mark.parametrize("hidden_size", [768, 2048])
    @pytest.mark.parametrize("rank", [16, 32])
    @pytest.mark.parametrize("num_experts", [8, 64])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_shrink_correctness(
        self, num_tokens, hidden_size, rank, num_experts, dtype
    ):
        from flashinfer.fused_moe.bgmv_moe import bgmv_moe_shrink, fill_w_ptr

        top_k = 2
        num_loras = 4
        num_slices = 1

        data = generate_test_data(
            num_tokens,
            hidden_size,
            rank,
            num_experts,
            top_k,
            num_loras,
            num_slices,
            dtype,
        )

        # Reference
        ref_out = reference_moe_bgmv_shrink(
            data["x"],
            data["lora_a_weights"],
            data["sorted_token_ids"],
            data["expert_ids"],
            data["lora_indices"],
        )

        # CUDA kernel
        num_pairs = data["num_pairs"]
        w_ptr = torch.zeros(num_slices, num_experts, dtype=torch.int64, device="cuda")
        lora_stride = fill_w_ptr(w_ptr, data["lora_a_weights"][0], num_experts, 0)

        cuda_out = torch.zeros(num_slices, num_pairs, rank, dtype=dtype, device="cuda")
        bgmv_moe_shrink(
            cuda_out,
            data["x"],
            w_ptr,
            data["sorted_token_ids"],
            data["expert_ids"],
            data["lora_indices"],
            lora_stride,
        )

        # Compare
        torch.testing.assert_close(
            cuda_out.float(),
            ref_out.float(),
            atol=1e-2,
            rtol=1e-2,
            msg=f"Shrink mismatch: tokens={num_tokens}, hidden={hidden_size}, "
            f"rank={rank}, experts={num_experts}, dtype={dtype}",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestBgmvMoeExpand:
    """Test the expand kernel against reference."""

    def setup_method(self):
        _skip_if_unsupported_sm()

    @pytest.mark.parametrize("num_tokens", [1, 4, 32])
    @pytest.mark.parametrize("hidden_size", [768, 2048])
    @pytest.mark.parametrize("rank", [16, 32])
    @pytest.mark.parametrize("num_experts", [8, 64])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_expand_correctness(
        self, num_tokens, hidden_size, rank, num_experts, dtype
    ):
        from flashinfer.fused_moe.bgmv_moe import bgmv_moe_expand, fill_w_ptr

        top_k = 2
        num_loras = 4
        num_slices = 1

        data = generate_test_data(
            num_tokens,
            hidden_size,
            rank,
            num_experts,
            top_k,
            num_loras,
            num_slices,
            dtype,
        )

        # Generate shrink output (use reference for isolation)
        shrink_out = reference_moe_bgmv_shrink(
            data["x"],
            data["lora_a_weights"],
            data["sorted_token_ids"],
            data["expert_ids"],
            data["lora_indices"],
        )

        # Reference expand
        ref_out = reference_moe_bgmv_expand(
            shrink_out,
            data["lora_b_weights"],
            data["sorted_token_ids"],
            data["expert_ids"],
            data["lora_indices"],
            data["topk_weights"],
            num_tokens,
        )

        # CUDA kernel
        w_ptr = torch.zeros(num_slices, num_experts, dtype=torch.int64, device="cuda")
        lora_stride = fill_w_ptr(w_ptr, data["lora_b_weights"][0], num_experts, 0)

        slice_start_loc = torch.zeros(num_slices, dtype=torch.int64, device="cuda")
        feat_out = hidden_size
        output_slices = [feat_out] * num_slices

        cuda_out = torch.zeros(num_tokens, feat_out, dtype=torch.float32, device="cuda")
        bgmv_moe_expand(
            cuda_out,
            shrink_out,
            w_ptr,
            data["sorted_token_ids"],
            data["expert_ids"],
            data["topk_weights"],
            data["lora_indices"],
            slice_start_loc,
            output_slices,
            lora_stride,
        )

        # Compare
        torch.testing.assert_close(
            cuda_out,
            ref_out,
            atol=1e-2,
            rtol=1e-2,
            msg=f"Expand mismatch: tokens={num_tokens}, hidden={hidden_size}, "
            f"rank={rank}, experts={num_experts}, dtype={dtype}",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestBgmvMoeEndToEnd:
    """End-to-end test: shrink + expand combined."""

    def setup_method(self):
        _skip_if_unsupported_sm()

    @pytest.mark.parametrize("num_tokens", [1, 8, 32])
    @pytest.mark.parametrize("hidden_size", [768, 2048])
    @pytest.mark.parametrize("rank", [16, 32])
    @pytest.mark.parametrize("num_experts", [8, 64])
    @pytest.mark.parametrize("top_k", [2])
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    def test_end_to_end(self, num_tokens, hidden_size, rank, num_experts, top_k, dtype):
        from flashinfer.fused_moe.bgmv_moe import bgmv_moe

        num_loras = 4
        num_slices = 1

        data = generate_test_data(
            num_tokens,
            hidden_size,
            rank,
            num_experts,
            top_k,
            num_loras,
            num_slices,
            dtype,
        )

        # Reference
        ref_out = reference_bgmv_moe(
            data["x"],
            data["lora_a_weights"],
            data["lora_b_weights"],
            data["sorted_token_ids"],
            data["expert_ids"],
            data["lora_indices"],
            data["topk_weights"],
        )

        # CUDA kernel (high-level API)
        cuda_out = bgmv_moe(
            data["x"],
            data["lora_a_weights"],
            data["lora_b_weights"],
            data["sorted_token_ids"],
            data["expert_ids"],
            data["lora_indices"],
            data["topk_weights"],
            num_experts,
        )

        # Compare
        torch.testing.assert_close(
            cuda_out.float(),
            ref_out.float(),
            atol=5e-2,
            rtol=5e-2,
            msg=f"E2E mismatch: tokens={num_tokens}, hidden={hidden_size}, "
            f"rank={rank}, experts={num_experts}, top_k={top_k}",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestBgmvMoeEdgeCases:
    """Edge case tests."""

    def setup_method(self):
        _skip_if_unsupported_sm()

    def test_all_tokens_no_lora(self):
        """All tokens have lora_id=-1, output should be zero."""
        from flashinfer.fused_moe.bgmv_moe import bgmv_moe

        num_tokens, hidden_size, rank, num_experts, top_k = 16, 768, 16, 8, 2
        dtype = torch.bfloat16
        num_loras = 4
        num_slices = 1

        data = generate_test_data(
            num_tokens,
            hidden_size,
            rank,
            num_experts,
            top_k,
            num_loras,
            num_slices,
            dtype,
        )
        # Set all lora_indices to -1
        data["lora_indices"].fill_(-1)

        out = bgmv_moe(
            data["x"],
            data["lora_a_weights"],
            data["lora_b_weights"],
            data["sorted_token_ids"],
            data["expert_ids"],
            data["lora_indices"],
            data["topk_weights"],
            num_experts,
        )

        assert torch.all(out == 0), "Output should be zero when no LoRA is active"

    def test_single_token_single_expert(self):
        """Minimal case: 1 token, 1 expert, 1 LoRA."""
        from flashinfer.fused_moe.bgmv_moe import bgmv_moe

        num_tokens, hidden_size, rank, num_experts, top_k = 1, 768, 8, 1, 1
        dtype = torch.bfloat16
        num_loras = 1
        num_slices = 1

        data = generate_test_data(
            num_tokens,
            hidden_size,
            rank,
            num_experts,
            top_k,
            num_loras,
            num_slices,
            dtype,
        )
        data["lora_indices"][0] = 0  # Ensure valid LoRA

        ref_out = reference_bgmv_moe(
            data["x"],
            data["lora_a_weights"],
            data["lora_b_weights"],
            data["sorted_token_ids"],
            data["expert_ids"],
            data["lora_indices"],
            data["topk_weights"],
        )

        cuda_out = bgmv_moe(
            data["x"],
            data["lora_a_weights"],
            data["lora_b_weights"],
            data["sorted_token_ids"],
            data["expert_ids"],
            data["lora_indices"],
            data["topk_weights"],
            num_experts,
        )

        torch.testing.assert_close(
            cuda_out.float(), ref_out.float(), atol=1e-2, rtol=1e-2
        )

    def test_multi_slice_w13(self):
        """Test with 2 slices (simulating gate+up projection)."""
        from flashinfer.fused_moe.bgmv_moe import bgmv_moe

        num_tokens, hidden_size, rank, num_experts, top_k = 8, 2048, 16, 8, 2
        dtype = torch.bfloat16
        num_loras = 4
        num_slices = 2

        data = generate_test_data(
            num_tokens,
            hidden_size,
            rank,
            num_experts,
            top_k,
            num_loras,
            num_slices,
            dtype,
        )

        ref_out = reference_bgmv_moe(
            data["x"],
            data["lora_a_weights"],
            data["lora_b_weights"],
            data["sorted_token_ids"],
            data["expert_ids"],
            data["lora_indices"],
            data["topk_weights"],
        )

        cuda_out = bgmv_moe(
            data["x"],
            data["lora_a_weights"],
            data["lora_b_weights"],
            data["sorted_token_ids"],
            data["expert_ids"],
            data["lora_indices"],
            data["topk_weights"],
            num_experts,
        )

        torch.testing.assert_close(
            cuda_out.float(), ref_out.float(), atol=5e-2, rtol=5e-2
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
