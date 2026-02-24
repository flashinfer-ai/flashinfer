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
Numerical accuracy tests for CuteDSL Fused MoE NVFP4 on Blackwell GPUs.

This test file covers both APIs:
1. Functional API: `cute_dsl_fused_moe_nvfp4`
2. Wrapper API: `CuteDslMoEWrapper`

Tests include:
- Numerical accuracy against reference implementation
- CUDA graph capture and replay
- Auto-tuning integration
- API consistency between functional and wrapper APIs
"""

import pytest
import torch
from torch.nn import functional as F

from flashinfer.cute_dsl import is_cute_dsl_available


def is_sm100_family():
    """Check for SM100 family (Blackwell: SM100, SM103, SM110).

    CuteDSL MoE NVFP4 kernels are optimized for SM100 architecture.
    SM120+ (Rubin) may have different shared memory/TMEM configurations.
    """
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    return props.major == 10


# Skip decorators
cute_dsl_available = pytest.mark.skipif(
    not is_cute_dsl_available(), reason="CuteDSL not available"
)
sm100_required = pytest.mark.skipif(
    not is_sm100_family(),
    reason="Requires SM100 family GPU (Blackwell: SM100, SM103, SM110)",
)


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU activation: x * sigmoid(x)"""
    return x * torch.sigmoid(x)


def interleave_linear_and_gate(
    x: torch.Tensor, group_size: int = 64, dim: int = -1
) -> torch.Tensor:
    """Interleave linear and gate weights for SwiGLU."""
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
    """Simulate FP4 quantization and dequantization for reference computation."""
    from flashinfer.fp4_quantization import fp4_quantize, e2m1_and_ufp8sf_scale_to_float

    tensor_bf16 = tensor.to(torch.bfloat16)
    fp4_packed, sf = fp4_quantize(
        tensor_bf16,
        global_scale=global_scale,
        sf_vec_size=sf_vec_size,
        is_sf_swizzled_layout=False,
    )

    sf_uint8 = sf.view(torch.uint8).reshape(-1)
    dequantized = e2m1_and_ufp8sf_scale_to_float(
        fp4_packed.cpu(),
        sf_uint8.cpu(),
        (1.0 / global_scale).cpu(),
        sf_vec_size=sf_vec_size,
        ufp8_type=1,
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
    num_local_experts: int = None,
    local_expert_offset: int = 0,
) -> torch.Tensor:
    """Compute reference MoE output using PyTorch operations on GPU.

    Args:
        hidden_states: Input hidden states [num_tokens, hidden_size]
        gemm1_weights: GEMM1 weights [num_local_experts, 2*intermediate_size, hidden_size]
        gemm2_weights: GEMM2 weights [num_local_experts, hidden_size, intermediate_size]
        token_selected_experts: Selected expert IDs (global) [num_tokens, top_k]
        token_final_scales: Routing weights [num_tokens, top_k]
        num_tokens: Number of tokens
        num_experts: Total number of experts (global)
        top_k: Number of experts per token
        hidden_size: Hidden dimension
        intermediate_size: Intermediate dimension
        fc2_input_scale: Optional scale for FC2 input quantization
        num_local_experts: Number of local experts (for EP). Defaults to num_experts.
        local_expert_offset: Starting expert ID for this EP rank. Defaults to 0.

    Returns:
        Output tensor [num_tokens, hidden_size]
    """
    if num_local_experts is None:
        num_local_experts = num_experts

    device = hidden_states.device

    hidden_states = hidden_states.float()
    gemm1_weights = gemm1_weights.float()
    gemm2_weights = gemm2_weights.float()

    output = torch.zeros((num_tokens, hidden_size), dtype=torch.float32, device=device)

    for token_idx in range(num_tokens):
        token_input = hidden_states[token_idx : token_idx + 1]

        for k in range(top_k):
            expert_idx = token_selected_experts[token_idx, k].item()
            scale = token_final_scales[token_idx, k].item()

            # Skip invalid expert IDs
            if expert_idx < 0 or expert_idx >= num_experts:
                continue

            # Convert global expert ID to local index for EP
            local_idx = expert_idx - local_expert_offset
            if local_idx < 0 or local_idx >= num_local_experts:
                # This expert is not on this EP rank, skip
                continue

            w1 = gemm1_weights[local_idx]
            gemm1_out = token_input @ w1.T

            linear = gemm1_out[:, :intermediate_size]
            gate = gemm1_out[:, intermediate_size:]
            swiglu_out = silu(gate) * linear

            if fc2_input_scale is not None:
                swiglu_out = quant_dequant_fp4_reference(
                    swiglu_out, fc2_input_scale, sf_vec_size=16
                )

            w2 = gemm2_weights[local_idx]
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
    """Check numerical accuracy with percentage-based tolerance."""
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


# =============================================================================
# Test Class: Functional API (cute_dsl_fused_moe_nvfp4)
# =============================================================================


@cute_dsl_available
@sm100_required
class TestCuteDslFusedMoeFunctional:
    """Tests for the functional API: cute_dsl_fused_moe_nvfp4."""

    @pytest.mark.parametrize(
        "hidden_size,intermediate_size", [(256, 512), (1024, 2048)]
    )
    @pytest.mark.parametrize("top_k", [1, 2, 8])
    @pytest.mark.parametrize("num_tokens", [128, 515, 1024])
    @pytest.mark.parametrize("num_experts", [256, 384])
    def test_numerical_accuracy(
        self,
        num_tokens: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
    ):
        """Accuracy test for functional API across configurations."""
        from flashinfer import cute_dsl_fused_moe_nvfp4

        num_local_experts = num_experts

        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            top_k=top_k,
        )

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
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

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

    def test_with_autotune(self):
        """Test functional API with autotune context."""
        from flashinfer import autotune
        from flashinfer import cute_dsl_fused_moe_nvfp4

        num_tokens, hidden_size, intermediate_size = 256, 256, 512
        num_experts, top_k = 256, 2

        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )

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


# =============================================================================
# Test Class: Wrapper API (CuteDslMoEWrapper)
# =============================================================================


@cute_dsl_available
@sm100_required
class TestCuteDslMoEWrapper:
    """Tests for the wrapper API: CuteDslMoEWrapper."""

    @pytest.mark.parametrize("num_tokens", [128, 256, 512])
    @pytest.mark.parametrize("top_k", [2, 8])
    @pytest.mark.parametrize("num_experts", [256, 384])
    def test_wrapper_accuracy(self, num_tokens: int, top_k: int, num_experts: int):
        """Accuracy test for wrapper API."""
        from flashinfer import CuteDslMoEWrapper

        hidden_size, intermediate_size = 256, 512

        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )

        # Create wrapper WITHOUT CUDA graph
        moe = CuteDslMoEWrapper(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            use_cuda_graph=False,
        )

        result = moe.run(
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
        )

        assert result.shape == (num_tokens, hidden_size)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

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

    @pytest.mark.parametrize("num_tokens", [64, 128, 256])
    @pytest.mark.parametrize("num_experts", [256, 384])
    def test_wrapper_cuda_graph(self, num_tokens: int, num_experts: int):
        """Test wrapper API with CUDA graph capture and replay."""
        from flashinfer import CuteDslMoEWrapper

        hidden_size, intermediate_size = 256, 512
        top_k = 2

        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )

        # Create wrapper WITH CUDA graph
        moe = CuteDslMoEWrapper(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            use_cuda_graph=True,
            max_num_tokens=num_tokens,
        )

        # Warmup
        for _ in range(3):
            moe.run(
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
            )
        torch.cuda.synchronize()

        # Capture CUDA graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            output = moe.run(
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
            )
        torch.cuda.synchronize()

        # Note: CUDA graph capture doesn't execute - output may be zeros here
        # Actual execution happens during replay
        assert output.shape == (num_tokens, hidden_size)

        # First replay to get actual output
        g.replay()
        torch.cuda.synchronize()

        # Verify output is valid after first replay
        assert not torch.isnan(output).any(), "NaN after first replay"
        assert not (output == 0).all(), "All zeros after first replay"

        # Test replay consistency (allow small numerical differences due to FP4 atomics)
        results = []
        for _ in range(3):
            g.replay()
            torch.cuda.synchronize()
            results.append(output.clone())

        # All replays should produce very similar results (small FP4 tolerance)
        for i in range(1, len(results)):
            max_diff = (results[0] - results[i]).abs().max().item()
            # FP4 atomics can have small non-determinism
            assert max_diff < 0.5, f"Replay {i} differs too much: max_diff={max_diff}"

        # Verify accuracy
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

        passed, percent_within, atol = check_accuracy(results[0], ref_output)
        assert passed, (
            f"CUDA graph accuracy: {percent_within * 100:.2f}% (atol={atol:.4f})"
        )

    def test_wrapper_with_autotune(self):
        """Test wrapper API with autotune context."""
        from flashinfer import autotune
        from flashinfer import CuteDslMoEWrapper

        num_tokens, hidden_size, intermediate_size = 256, 256, 512
        num_experts, top_k = 256, 2

        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )

        moe = CuteDslMoEWrapper(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            use_cuda_graph=False,
        )

        with autotune(True):
            result = moe.run(
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


# =============================================================================
# Test Class: API Consistency
# =============================================================================


@cute_dsl_available
@sm100_required
class TestApiConsistency:
    """Tests verifying consistency between functional and wrapper APIs."""

    def test_functional_vs_wrapper_output(self):
        """Verify functional and wrapper APIs produce the same output."""
        from flashinfer import CuteDslMoEWrapper, cute_dsl_fused_moe_nvfp4

        num_tokens, hidden_size, intermediate_size = 128, 256, 512
        num_experts, top_k = 256, 2

        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )

        # Functional API
        result_functional = cute_dsl_fused_moe_nvfp4(
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

        # Wrapper API
        moe = CuteDslMoEWrapper(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            use_cuda_graph=False,
        )

        result_wrapper = moe.run(
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
        )

        # Both should produce valid outputs
        assert result_functional.shape == result_wrapper.shape
        assert not torch.isnan(result_functional).any()
        assert not torch.isnan(result_wrapper).any()

        # Outputs should be very close (may not be exactly equal due to different
        # tuning paths, but should be within FP4 tolerance)
        diff = (result_functional - result_wrapper).abs()
        max_diff = diff.max().item()
        # Allow small differences from autotuner path differences
        assert max_diff < 1e-3, f"Max diff between APIs: {max_diff}"


# =============================================================================
# Test Class: Expert Parallelism
# =============================================================================


@cute_dsl_available
@sm100_required
class TestExpertParallelism:
    """Tests for expert parallelism (EP) configurations."""

    @pytest.mark.parametrize("ep_size", [1, 8, 32])
    @pytest.mark.parametrize("ep_rank", [0, -1])  # -1 means last rank
    def test_wrapper_with_ep(self, ep_size: int, ep_rank: int):
        """Test wrapper API with expert parallelism and numerical accuracy.

        Tests different EP ranks to ensure local_expert_offset handling is correct.
        ep_rank=-1 is converted to the last rank (ep_size-1) to test non-zero offsets.
        """
        from flashinfer import CuteDslMoEWrapper

        # Convert -1 to last rank
        if ep_rank == -1:
            ep_rank = ep_size - 1

        num_tokens, hidden_size, intermediate_size = 256, 256, 512
        num_experts, top_k = 256, 8
        num_local_experts = num_experts // ep_size
        local_expert_offset = ep_rank * num_local_experts

        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            top_k=top_k,
        )

        # Keep original routing - the kernel should handle filtering
        # based on local_expert_offset and num_local_experts
        token_selected_experts = tensors["token_selected_experts"].clone()

        moe = CuteDslMoEWrapper(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_local_experts=num_local_experts,
            local_expert_offset=local_expert_offset,
        )

        result = moe.run(
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
        )

        assert result.shape == (num_tokens, hidden_size)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

        # Numerical accuracy verification against reference
        ref_output = compute_reference_moe_fp4(
            hidden_states=tensors["x_bf16"].float().cuda(),
            gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
            gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=token_selected_experts,
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=tensors["fc2_input_scale"],
            num_local_experts=num_local_experts,
            local_expert_offset=local_expert_offset,
        )

        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, (
            f"EP accuracy test failed (ep_size={ep_size}, ep_rank={ep_rank}, "
            f"offset={local_expert_offset}): {percent_within * 100:.2f}% within tolerance (atol={atol:.4f})"
        )

    @pytest.mark.parametrize("ep_size", [8])
    def test_functional_with_ep(self, ep_size: int):
        """Test functional API with expert parallelism and numerical accuracy."""
        from flashinfer import cute_dsl_fused_moe_nvfp4

        # Test middle rank to ensure offset handling works
        ep_rank = ep_size // 2

        num_tokens, hidden_size, intermediate_size = 256, 256, 512
        num_experts, top_k = 256, 8
        num_local_experts = num_experts // ep_size
        local_expert_offset = ep_rank * num_local_experts

        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            top_k=top_k,
        )

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
            local_expert_offset=local_expert_offset,
        )

        assert result.shape == (num_tokens, hidden_size)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

        # Numerical accuracy verification
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
            num_local_experts=num_local_experts,
            local_expert_offset=local_expert_offset,
        )

        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, (
            f"EP functional API accuracy test failed (ep_size={ep_size}, ep_rank={ep_rank}): "
            f"{percent_within * 100:.2f}% within tolerance (atol={atol:.4f})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
