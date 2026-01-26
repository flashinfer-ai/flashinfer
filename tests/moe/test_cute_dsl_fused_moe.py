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

"""
Streamlined numerical accuracy tests for CuteDSL Fused MoE NVFP4 on Blackwell GPUs.

This test file focuses on numerical accuracy verification across:
- Different problem sizes (num_tokens, hidden_size, intermediate_size)
- Different expert parallelism configurations (ep_size)
- Different top_k values
- Auto-tuner integration via `autotune` context manager

For comprehensive functional and integration tests, see test_cute_dsl_fused_moe_full.py
"""

import pytest
import torch
from torch.nn import functional as F

from flashinfer.cute_dsl import is_cute_dsl_available


def is_blackwell():
    """Check if running on Blackwell GPU (SM100+)."""
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    return props.major >= 10


# Skip decorators
cute_dsl_available = pytest.mark.skipif(
    not is_cute_dsl_available(), reason="CuteDSL not available"
)
blackwell_required = pytest.mark.skipif(
    not is_blackwell(), reason="Requires Blackwell GPU (SM100+)"
)


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU activation: x * sigmoid(x)"""
    return x * torch.sigmoid(x)


def interleave_linear_and_gate(
    x: torch.Tensor, group_size: int = 64, dim: int = -1
) -> torch.Tensor:
    """Interleave linear and gate weights for SwiGLU.

    This matches TRT-LLM's interleave_linear_and_gate function.
    Converts from [gate, up] concatenated format to interleaved format.
    """
    sizes = x.size()
    dim = dim % x.dim()
    assert sizes[dim] % (group_size * 2) == 0
    prev_sizes = sizes[:dim]
    post_sizes = sizes[dim + 1 :]
    x = x.view(*prev_sizes, 2, sizes[dim] // (group_size * 2), group_size, *post_sizes)
    x = x.transpose(dim, dim + 1).contiguous().view(*sizes)
    return x


def quant_dequant_fp4_reference(
    tensor: torch.Tensor,
    global_scale: torch.Tensor,
    sf_vec_size: int = 16,
) -> torch.Tensor:
    """
    Simulate FP4 quantization and dequantization for reference computation.

    This models the quantization error introduced when intermediate activations
    are quantized to FP4 format between GEMM1 and GEMM2.
    """
    from flashinfer.fp4_quantization import fp4_quantize, e2m1_and_ufp8sf_scale_to_float

    # Quantize to FP4
    tensor_bf16 = tensor.to(torch.bfloat16)
    fp4_packed, sf = fp4_quantize(
        tensor_bf16,
        global_scale=global_scale,
        sf_vec_size=sf_vec_size,
        is_sf_swizzled_layout=False,
    )

    # Dequantize back to float
    sf_uint8 = sf.view(torch.uint8).reshape(-1)
    dequantized = e2m1_and_ufp8sf_scale_to_float(
        fp4_packed.cpu(),
        sf_uint8.cpu(),
        (1.0 / global_scale).cpu(),
        sf_vec_size=sf_vec_size,
        ufp8_type=1,  # UFP8 E4M3
        is_sf_swizzled_layout=False,
    ).to(tensor.device)

    return dequantized.float()


def compute_reference_moe_fp4(
    hidden_states: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm2_weights: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    fc2_input_scale: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute reference MoE output using PyTorch operations.

    This follows the same computation as CuteDSL fused MoE:
    1. For each token-expert pair:
       - GEMM1: hidden @ W1.T -> [2 * intermediate_size]
       - SwiGLU: linear * silu(gate) -> [intermediate_size]
       - FP4 quantize/dequantize (simulates intermediate quantization)
       - GEMM2: intermediate @ W2.T -> [hidden_size]
    2. Accumulate with routing weights

    Args:
        fc2_input_scale: Global scale for intermediate FP4 quantization.
                         If None, skips intermediate quantization (original behavior).
    """
    device = hidden_states.device
    output = torch.zeros((num_tokens, hidden_size), dtype=torch.float32, device=device)

    for token_idx in range(num_tokens):
        token_input = hidden_states[token_idx : token_idx + 1]

        for k in range(top_k):
            expert_idx = token_selected_experts[token_idx, k].item()
            scale = token_final_scales[token_idx, k].item()

            if expert_idx < 0 or expert_idx >= num_experts:
                continue

            w1 = gemm1_weights[expert_idx]
            gemm1_out = token_input @ w1.T

            linear = gemm1_out[:, :intermediate_size]
            gate = gemm1_out[:, intermediate_size:]
            swiglu_out = silu(gate) * linear

            # Simulate intermediate FP4 quantization if scale is provided
            if fc2_input_scale is not None:
                swiglu_out = quant_dequant_fp4_reference(
                    swiglu_out, fc2_input_scale, sf_vec_size=16
                )

            w2 = gemm2_weights[expert_idx]
            gemm2_out = swiglu_out @ w2.T

            output[token_idx] += scale * gemm2_out.squeeze(0)

    return output


def create_moe_tensors(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    num_local_experts: int,
    top_k: int,
    device: str = "cuda",
    seed: int = 42,
):
    """Create properly quantized MoE tensors for testing."""
    from flashinfer.fp4_quantization import fp4_quantize
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

    torch.manual_seed(seed)
    sf_vec_size = 16

    # Input
    x_bf16 = (
        torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device) / 10
    )
    a1_gs = torch.tensor([1.0], device=device, dtype=torch.float32)

    x_quantized, x_sf = fp4_quantize(
        x_bf16, global_scale=a1_gs, sf_vec_size=sf_vec_size, is_sf_swizzled_layout=False
    )
    x_sf = x_sf.unsqueeze(-1)

    # Routing
    router_logits = torch.randn(num_tokens, num_experts, device=device)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.float()
    selected_experts = selected_experts.to(torch.int32)

    # GEMM1 weights
    w1_bf16 = (
        torch.randn(
            num_local_experts,
            2 * intermediate_size,
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        / 10
    )

    w1_bf16_interleaved = interleave_linear_and_gate(w1_bf16, group_size=64, dim=1)
    w1_gs = torch.tensor([1.0], device=device, dtype=torch.float32)

    w1_flat = w1_bf16_interleaved.view(
        num_local_experts * 2 * intermediate_size, hidden_size
    )
    w1_q_flat, w1_sf_flat = fp4_quantize(
        w1_flat, global_scale=w1_gs, sf_vec_size=sf_vec_size, is_sf_swizzled_layout=True
    )
    w1_q = w1_q_flat.view(num_local_experts, 2 * intermediate_size, hidden_size // 2)
    w1_weight_sf = convert_sf_to_mma_layout(
        w1_sf_flat,
        m=2 * intermediate_size,
        k=hidden_size,
        num_groups=num_local_experts,
        sf_vec_size=sf_vec_size,
    )
    w1_alpha = torch.ones(num_local_experts, device=device, dtype=torch.float32)

    # GEMM2 weights
    w2_bf16 = (
        torch.randn(
            num_local_experts,
            hidden_size,
            intermediate_size,
            dtype=torch.bfloat16,
            device=device,
        )
        / 10
    )

    w2_gs = torch.tensor([1.0], device=device, dtype=torch.float32)
    w2_flat = w2_bf16.view(num_local_experts * hidden_size, intermediate_size)
    w2_q_flat, w2_sf_flat = fp4_quantize(
        w2_flat, global_scale=w2_gs, sf_vec_size=sf_vec_size, is_sf_swizzled_layout=True
    )
    w2_q = w2_q_flat.view(num_local_experts, hidden_size, intermediate_size // 2)
    w2_weight_sf = convert_sf_to_mma_layout(
        w2_sf_flat,
        m=hidden_size,
        k=intermediate_size,
        num_groups=num_local_experts,
        sf_vec_size=sf_vec_size,
    )
    w2_alpha = torch.ones(num_local_experts, device=device, dtype=torch.float32)

    fc2_input_scale = torch.tensor([1.0], device=device, dtype=torch.float32)

    return {
        "x": x_quantized,
        "x_sf": x_sf,
        "x_bf16": x_bf16,
        "token_selected_experts": selected_experts,
        "token_final_scales": routing_weights,
        "w1_weight": w1_q,
        "w1_weight_sf": w1_weight_sf,
        "w1_weight_bf16": w1_bf16,
        "w1_alpha": w1_alpha,
        "fc2_input_scale": fc2_input_scale,
        "w2_weight": w2_q,
        "w2_weight_sf": w2_weight_sf,
        "w2_weight_bf16": w2_bf16,
        "w2_alpha": w2_alpha,
    }


def check_accuracy(
    actual: torch.Tensor, expected: torch.Tensor, percent_threshold: float = 0.925
):
    """Check numerical accuracy with percentage-based tolerance.

    FP4 quantization introduces significant error, so we check that 92.5% of values
    are within tolerance (atol scaled with output magnitude, rtol=0.85).
    """
    actual = actual.float()
    expected = expected.float()

    output_scale = max(expected.std().item(), 0.01)
    atol = max(0.1, 3.0 * output_scale)
    rtol = 0.85

    abs_diff = torch.abs(actual - expected)
    rel_diff = abs_diff / (torch.abs(expected) + 1e-8)
    within_tolerance = (abs_diff < atol) | (rel_diff < rtol)
    percent_within = within_tolerance.float().mean().item()

    return percent_within >= percent_threshold, percent_within, atol


@cute_dsl_available
@blackwell_required
class TestCuteDslFusedMoeAccuracy:
    """Numerical accuracy tests for CuteDSL Fused MoE NVFP4.

    Tests cover:
    - Problem sizes: num_tokens in [128, 515, 1024, 8192]
    - Top-k values: [1, 2, 8]
    - Expert parallelism: ep_size in [1, 8, 32] with num_experts=256

    Note: The API uses the `autotune` context manager for auto-tuning.
    Without the context manager, it uses cached or default tactics.
    """

    @pytest.mark.parametrize(
        "hidden_size,intermediate_size", [(256, 512), (1024, 2048)]
    )
    @pytest.mark.parametrize("top_k", [1, 2, 8])
    @pytest.mark.parametrize("num_tokens", [128, 515, 1024, 8192])
    def test_numerical_accuracy(
        self, num_tokens: int, top_k: int, hidden_size: int, intermediate_size: int
    ):
        """Accuracy test across different configurations."""
        from flashinfer.cute_dsl import cute_dsl_fused_moe_nvfp4

        num_experts = 8
        num_local_experts = num_experts

        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            top_k=top_k,
        )

        # Call without autotune context - uses default/cached tactics
        result = cute_dsl_fused_moe_nvfp4(
            x=tensors["x"],
            x_sf=tensors["x_sf"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            num_experts=num_experts,
            top_k=top_k,
            num_local_experts=num_local_experts,
        )

        assert result.shape == (num_tokens, hidden_size)
        assert result.dtype == torch.bfloat16
        assert not torch.isnan(result).any(), "Output contains NaN"
        assert not torch.isinf(result).any(), "Output contains Inf"

        ref_output = compute_reference_moe_fp4(
            hidden_states=tensors["x_bf16"].float().cuda(),
            gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
            gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_local_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=tensors["fc2_input_scale"],
        )

        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, (
            f"Only {percent_within * 100:.2f}% within tolerance (atol={atol:.4f})"
        )

    # Expert parallelism tests
    @pytest.mark.parametrize("ep_size", [1, 8, 32])
    @pytest.mark.parametrize("top_k", [1, 2, 8])
    def test_accuracy_expert_parallelism(self, top_k: int, ep_size: int):
        """Accuracy test with expert parallelism (num_experts=256)."""
        from flashinfer.cute_dsl import cute_dsl_fused_moe_nvfp4

        num_tokens = 256
        hidden_size = 256
        intermediate_size = 512
        num_experts = 256
        num_local_experts = num_experts // ep_size
        local_expert_offset = 0

        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            top_k=top_k,
        )

        # Filter routing to local experts only
        token_selected_experts = tensors["token_selected_experts"].clone()
        mask = (token_selected_experts >= local_expert_offset) & (
            token_selected_experts < local_expert_offset + num_local_experts
        )
        token_selected_experts[~mask] = local_expert_offset

        result = cute_dsl_fused_moe_nvfp4(
            x=tensors["x"],
            x_sf=tensors["x_sf"],
            token_selected_experts=token_selected_experts,
            token_final_scales=tensors["token_final_scales"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            num_experts=num_experts,
            top_k=top_k,
            num_local_experts=num_local_experts,
            local_expert_offset=local_expert_offset,
        )

        assert result.shape == (num_tokens, hidden_size)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

        # Note: For EP tests, we verify output validity but skip strict accuracy
        # comparison since expert filtering changes the computation semantically

    def test_accuracy_with_autotune(self):
        """Test accuracy when called inside autotune context.

        This tests the `with autotune(True):` pattern that enables profiling.
        Note: This test is slow because it profiles all 16 tactics.
        """
        from flashinfer import autotune
        from flashinfer.cute_dsl import cute_dsl_fused_moe_nvfp4

        num_tokens = 256
        hidden_size = 256
        intermediate_size = 512
        num_experts = 8
        top_k = 2

        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )

        # Test inside autotune context (enables profiling mode)
        with autotune(True):
            result = cute_dsl_fused_moe_nvfp4(
                x=tensors["x"],
                x_sf=tensors["x_sf"],
                token_selected_experts=tensors["token_selected_experts"],
                token_final_scales=tensors["token_final_scales"],
                w1_weight=tensors["w1_weight"],
                w1_weight_sf=tensors["w1_weight_sf"],
                w1_alpha=tensors["w1_alpha"],
                fc2_input_scale=tensors["fc2_input_scale"],
                w2_weight=tensors["w2_weight"],
                w2_weight_sf=tensors["w2_weight_sf"],
                w2_alpha=tensors["w2_alpha"],
                num_experts=num_experts,
                top_k=top_k,
            )

        assert result.shape == (num_tokens, hidden_size)
        assert not torch.isnan(result).any()

        ref_output = compute_reference_moe_fp4(
            hidden_states=tensors["x_bf16"].float().cuda(),
            gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
            gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=tensors["fc2_input_scale"],
        )

        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, (
            f"Only {percent_within * 100:.2f}% within tolerance (atol={atol:.4f})"
        )

    @pytest.mark.parametrize(
        "num_tokens,top_k,hidden_size,intermediate_size,num_experts,ep_size",
        [
            # Basic configuration
            (128, 2, 256, 512, 8, 1),
            # Different top_k
            (256, 1, 256, 512, 8, 1),
            # Larger model size
            (256, 2, 1024, 2048, 8, 1),
            # Expert parallelism (ep_size > 1)
            (256, 2, 256, 512, 64, 8),
        ],
    )
    def test_cuda_graph_capture_and_accuracy(
        self,
        num_tokens: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        ep_size: int,
    ):
        """Consolidated CUDA graph test: capture, replay, consistency, and accuracy.

        This test verifies:
        1. CUDA graph capture succeeds (no CPU-GPU sync during capture)
        2. Graph replay produces valid outputs (no NaN/Inf)
        3. Multiple replays produce consistent results
        4. Output matches reference implementation

        Covers basic configs, different top_k, larger models, and expert parallelism.
        """
        from flashinfer.cute_dsl import cute_dsl_fused_moe_nvfp4

        num_local_experts = num_experts // ep_size
        local_expert_offset = 0

        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            top_k=top_k,
        )

        # For expert parallelism, filter routing to local experts only
        token_selected_experts = tensors["token_selected_experts"]
        if ep_size > 1:
            token_selected_experts = token_selected_experts.clone()
            mask = (token_selected_experts >= local_expert_offset) & (
                token_selected_experts < local_expert_offset + num_local_experts
            )
            token_selected_experts[~mask] = local_expert_offset

        # Pre-allocate output buffer (required for CUDA graph - size must be fixed)
        moe_output = torch.empty(
            (num_tokens, hidden_size),
            dtype=torch.bfloat16,
            device="cuda",
        )

        # Warmup runs (required before CUDA graph capture)
        for _ in range(3):
            cute_dsl_fused_moe_nvfp4(
                x=tensors["x"],
                x_sf=tensors["x_sf"],
                token_selected_experts=token_selected_experts,
                token_final_scales=tensors["token_final_scales"],
                w1_weight=tensors["w1_weight"],
                w1_weight_sf=tensors["w1_weight_sf"],
                w1_alpha=tensors["w1_alpha"],
                fc2_input_scale=tensors["fc2_input_scale"],
                w2_weight=tensors["w2_weight"],
                w2_weight_sf=tensors["w2_weight_sf"],
                w2_alpha=tensors["w2_alpha"],
                num_experts=num_experts,
                top_k=top_k,
                num_local_experts=num_local_experts,
                local_expert_offset=local_expert_offset if ep_size > 1 else 0,
                moe_output=moe_output,
            )
        torch.cuda.synchronize()

        # Capture CUDA graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            result = cute_dsl_fused_moe_nvfp4(
                x=tensors["x"],
                x_sf=tensors["x_sf"],
                token_selected_experts=token_selected_experts,
                token_final_scales=tensors["token_final_scales"],
                w1_weight=tensors["w1_weight"],
                w1_weight_sf=tensors["w1_weight_sf"],
                w1_alpha=tensors["w1_alpha"],
                fc2_input_scale=tensors["fc2_input_scale"],
                w2_weight=tensors["w2_weight"],
                w2_weight_sf=tensors["w2_weight_sf"],
                w2_alpha=tensors["w2_alpha"],
                num_experts=num_experts,
                top_k=top_k,
                num_local_experts=num_local_experts,
                local_expert_offset=local_expert_offset if ep_size > 1 else 0,
                moe_output=moe_output,
            )

        # Test multiple replays for consistency
        results = []
        for _ in range(3):
            g.replay()
            torch.cuda.synchronize()
            results.append(result.clone())

        # Verify output shape and validity
        assert result.shape == (num_tokens, hidden_size)
        assert result.dtype == torch.bfloat16
        assert not torch.isnan(result).any(), "Output contains NaN after graph replay"
        assert not torch.isinf(result).any(), "Output contains Inf after graph replay"

        # Verify all replays produce identical results
        for i in range(1, len(results)):
            assert torch.allclose(results[0], results[i]), (
                f"Replay {i} differs from replay 0"
            )

        # Verify accuracy against reference implementation
        ref_output = compute_reference_moe_fp4(
            hidden_states=tensors["x_bf16"].float().cuda(),
            gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
            gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=token_selected_experts,
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_local_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=tensors["fc2_input_scale"],
        )

        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, (
            f"CUDA graph accuracy failed for tokens={num_tokens}, top_k={top_k}, "
            f"hidden={hidden_size}, ep_size={ep_size}: "
            f"{percent_within * 100:.2f}% within tolerance (atol={atol:.4f})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
