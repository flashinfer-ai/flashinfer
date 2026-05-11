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
    """Check for SM100 family (Blackwell: SM100, SM103).

    CuteDSL MoE NVFP4 kernels are optimized for SM10x architecture.
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
    actual: torch.Tensor, expected: torch.Tensor, percent_threshold: float = 0.97
):
    """Check numerical accuracy with percentage-based tolerance.

    Tolerances are scaled by output magnitude to account for FP4 quantization
    noise growing with larger hidden dimensions.
    """
    actual = actual.float()
    expected = expected.float()

    output_scale = max(expected.std().item(), 0.01)
    atol = max(0.05, 1.5 * output_scale)
    rtol = 0.5

    abs_diff = torch.abs(actual - expected)
    rel_diff = abs_diff / (torch.abs(expected) + 1e-8)
    within_tolerance = (abs_diff < atol) | (rel_diff < rtol)
    percent_within = within_tolerance.float().mean().item()

    return percent_within >= percent_threshold, percent_within, atol


# =============================================================================
# Test Class: Tactic-enumeration structural invariants (no GPU required)
# =============================================================================


@cute_dsl_available
class TestTacticEnumeration:
    """Structural invariants for the tactic-enumeration helpers in
    flashinfer.fused_moe.cute_dsl.tuner.

    These tests run without a GPU. They exercise the enumeration
    functions directly to enforce invariants that the end-to-end
    accuracy tests can fail to detect when a tile size is gated out of
    ALL_MOE_TACTICS as a workaround.

    The MoE pipeline runs gemm1 (gather + SwiGLU) followed by gemm2
    (finalize fusion) back-to-back on the same padded token sequence.
    For the layouts to match, gemm1 and gemm2 must share the same
    mma_tiler M dimension and the same cluster_shape M dimension.
    """

    @pytest.mark.parametrize("tile_size", [128, 256])
    def test_gemm1_tactics_match_tile_size(self, tile_size):
        """Every gemm1 tactic must have mma_tiler[0] == tile_size and
        cluster_shape[0] == tile_size // 128 (1-CTA at tile=128, 2-CTA
        at tile=256)."""
        from flashinfer.fused_moe.cute_dsl.tuner import get_gemm1_valid_tactics

        tactics = get_gemm1_valid_tactics(tile_size)
        assert len(tactics) > 0, f"no gemm1 tactics returned at tile_size={tile_size}"
        expected_cluster_m = tile_size // 128
        for mma_tiler_mn, cluster_shape_mn, _ in tactics:
            assert mma_tiler_mn[0] == tile_size, (
                f"gemm1 mma_tiler[0]={mma_tiler_mn[0]} does not match "
                f"tile_size={tile_size}; tactic={(mma_tiler_mn, cluster_shape_mn)}"
            )
            assert cluster_shape_mn[0] == expected_cluster_m, (
                f"gemm1 cluster_shape[0]={cluster_shape_mn[0]} does not "
                f"match tile_size//128={expected_cluster_m}; "
                f"tactic={(mma_tiler_mn, cluster_shape_mn)}"
            )

    @pytest.mark.parametrize("tile_size", [128, 256])
    def test_gemm2_tactics_match_tile_size(self, tile_size):
        """Every gemm2 tactic must have mma_tiler[0] == tile_size and
        cluster_shape[0] == tile_size // 128. The finalize kernel
        consumes the upstream gemm1 output layout — a 1-CTA gemm2
        tactic at tile_size=256 cannot consume a 2-CTA gemm1 output
        and produces incorrect results (regression for #3067)."""
        from flashinfer.fused_moe.cute_dsl.tuner import get_gemm2_valid_tactics

        tactics = get_gemm2_valid_tactics(tile_size)
        assert len(tactics) > 0, f"no gemm2 tactics returned at tile_size={tile_size}"
        expected_cluster_m = tile_size // 128
        for mma_tiler_mn, cluster_shape_mn, _ in tactics:
            assert mma_tiler_mn[0] == tile_size, (
                f"gemm2 mma_tiler[0]={mma_tiler_mn[0]} does not match "
                f"tile_size={tile_size}; tactic={(mma_tiler_mn, cluster_shape_mn)}"
            )
            assert cluster_shape_mn[0] == expected_cluster_m, (
                f"gemm2 cluster_shape[0]={cluster_shape_mn[0]} does not "
                f"match tile_size//128={expected_cluster_m}; "
                f"tactic={(mma_tiler_mn, cluster_shape_mn)}"
            )

    def test_all_moe_tactics_pair_gemm1_and_gemm2_consistently(self):
        """Every (tile_size, gemm1_tactic, gemm2_tactic) tuple in
        ALL_MOE_TACTICS must have gemm1 and gemm2 share both
        mma_tiler[0] and cluster_shape[0] (the M dimensions). This
        catches a class of bug where the product loop in
        get_moe_valid_tactics accidentally pairs incompatible
        gemm1/gemm2 tactics, even if each individual enumeration is
        internally consistent."""
        from flashinfer.fused_moe.cute_dsl.tuner import ALL_MOE_TACTICS

        assert len(ALL_MOE_TACTICS) > 0
        for tile_size, gemm1_tactic, gemm2_tactic in ALL_MOE_TACTICS:
            gemm1_mma_m = gemm1_tactic[0][0]
            gemm1_cluster_m = gemm1_tactic[1][0]
            gemm2_mma_m = gemm2_tactic[0][0]
            gemm2_cluster_m = gemm2_tactic[1][0]
            assert gemm1_mma_m == gemm2_mma_m == tile_size, (
                f"gemm1/gemm2 mma_m mismatch in ALL_MOE_TACTICS at "
                f"tile_size={tile_size}: gemm1_mma_m={gemm1_mma_m}, "
                f"gemm2_mma_m={gemm2_mma_m}"
            )
            assert gemm1_cluster_m == gemm2_cluster_m == tile_size // 128, (
                f"gemm1/gemm2 cluster_m mismatch in ALL_MOE_TACTICS at "
                f"tile_size={tile_size}: gemm1_cluster_m={gemm1_cluster_m}, "
                f"gemm2_cluster_m={gemm2_cluster_m}"
            )


# =============================================================================
# Test Class: CuteDslMoEInputsHelper.inputs_pre_hook layout contract
# (no GPU required)
# =============================================================================


@cute_dsl_available
class TestInputsHelperContract:
    """Structural invariants for ``CuteDslMoEInputsHelper.inputs_pre_hook``.

    Tests run without a GPU. They exercise the cross-file contract that
    the hook's unpacking pattern (``x, x_sf, tse, *rest = inputs``) must
    match the input list layout produced by ``CuteDslMoEWrapper.run`` so
    that autotune profile inputs are not silently corrupted by a
    refactor that reorders the wrapper's inputs list.
    """

    def _build_synthetic_inputs(self, num_tokens: int, num_local_experts: int):
        """Mirror ``CuteDslMoEWrapper.run``'s inputs-list layout with
        small-but-shape-faithful tensors so the test runs in <1s on CPU."""
        n = num_tokens
        # Small dimensions for fast CPU allocation; sizes only matter for
        # shape checks, not numerical results.
        hidden = 128
        intermediate = 64
        top_k = 8
        sf_vec = 16
        return [
            torch.zeros(n, hidden // 2, dtype=torch.uint8),  # 0: x
            torch.zeros(n, hidden // sf_vec, dtype=torch.uint8),  # 1: x_sf
            torch.zeros(n, top_k, dtype=torch.int32),  # 2: token_selected_experts
            torch.zeros(n, top_k, dtype=torch.float32),  # 3: token_final_scales
            torch.zeros(
                num_local_experts, 2 * intermediate, hidden // 2, dtype=torch.uint8
            ),  # 4: w1_weight
            torch.zeros(
                num_local_experts, 2 * intermediate, hidden // sf_vec, dtype=torch.uint8
            ),  # 5: w1_weight_sf
            torch.zeros(num_local_experts, dtype=torch.float32),  # 6: w1_alpha
            torch.zeros(num_local_experts, dtype=torch.float32),  # 7: fc2_input_scale
            torch.zeros(
                num_local_experts, hidden, intermediate // 2, dtype=torch.uint8
            ),  # 8: w2_weight
            torch.zeros(
                num_local_experts, hidden, intermediate // sf_vec, dtype=torch.uint8
            ),  # 9: w2_weight_sf
            torch.zeros(num_local_experts, dtype=torch.float32),  # 10: w2_alpha
            torch.zeros(n, hidden, dtype=torch.bfloat16),  # 11: moe_output
        ]

    def test_hook_replaces_input_2_and_passes_through_rest(self):
        """``inputs_pre_hook`` must replace ``inputs[2]``
        (token_selected_experts) and pass through every other input
        unchanged. Pins the contract with ``CuteDslMoEWrapper.run`` —
        if someone reorders the inputs list (e.g. moves x_sf or a
        weight tensor) without updating the helper's unpacking, the
        autotune profile silently corrupts a different tensor."""
        from flashinfer.fused_moe.cute_dsl._inputs_helper import (
            CuteDslMoEInputsHelper,
        )

        num_local_experts = 16
        helper = CuteDslMoEInputsHelper(
            num_experts=256,
            top_k=8,
            num_local_experts=num_local_experts,
            local_expert_offset=0,
        )

        inputs = self._build_synthetic_inputs(
            num_tokens=64, num_local_experts=num_local_experts
        )
        original_tse = inputs[2]

        output = helper.inputs_pre_hook(inputs)

        assert len(output) == 12, f"Expected 12 outputs, got {len(output)}"
        # Index 2 must be replaced (different object identity), with
        # the same shape and dtype.
        assert output[2] is not original_tse, (
            "inputs[2] (token_selected_experts) must be REPLACED by the hook, "
            "not passed through. Hook implementation in _inputs_helper.py "
            "must build a fresh tensor for input #2."
        )
        assert output[2].shape == original_tse.shape, (
            f"Replaced tse has shape {output[2].shape}, expected {original_tse.shape}"
        )
        assert output[2].dtype == original_tse.dtype, (
            f"Replaced tse has dtype {output[2].dtype}, expected {original_tse.dtype}"
        )
        # Every other input MUST pass through with object identity preserved.
        # If this breaks, the hook is mutating something it shouldn't, OR the
        # wrapper's inputs-list ordering has drifted from the hook's unpacking.
        for i in (0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11):
            assert output[i] is inputs[i], (
                f"inputs[{i}] must pass through the hook unchanged (object identity). "
                f"This typically indicates the inputs-list ordering in "
                f"CuteDslMoEWrapper.run has drifted from the hook's unpacking pattern "
                f"in CuteDslMoEInputsHelper.inputs_pre_hook."
            )

    def test_hook_is_deterministic_across_helper_instances(self):
        """Two ``CuteDslMoEInputsHelper`` instances with the same seed
        must produce tensor-equal replacement ``token_selected_experts``
        for the same input. This is the core determinism property the
        helper provides — cross-process autotune-pick variance is
        eliminated only if seeded sampling is deterministic."""
        from flashinfer.fused_moe.cute_dsl._inputs_helper import (
            CuteDslMoEInputsHelper,
        )

        helper_a = CuteDslMoEInputsHelper(
            num_experts=256, top_k=8, num_local_experts=16, local_expert_offset=0
        )
        helper_b = CuteDslMoEInputsHelper(
            num_experts=256, top_k=8, num_local_experts=16, local_expert_offset=0
        )

        inputs_a = self._build_synthetic_inputs(num_tokens=64, num_local_experts=16)
        inputs_b = self._build_synthetic_inputs(num_tokens=64, num_local_experts=16)

        out_a = helper_a.inputs_pre_hook(inputs_a)
        out_b = helper_b.inputs_pre_hook(inputs_b)

        assert torch.equal(out_a[2], out_b[2]), (
            "Two helpers with the same seed produced different "
            "token_selected_experts tensors. The seeded "
            "torch.random.fork_rng + manual_seed pattern in "
            "generate_token_selected_experts is broken."
        )


# =============================================================================
# Test Class: get_max_num_tiles / get_max_num_permuted_tokens (no GPU required)
# =============================================================================


@cute_dsl_available
class TestGetMaxNumTiles:
    """Worst-case upper-bound tests for
    ``flashinfer.fused_moe.cute_dsl.moe_utils.get_max_num_tiles``.

    The function must return the tight upper bound on
    ``sum_e ceil(K_e / tile_size)``, where ``K_e`` is the per-local-expert
    token count and ``sum_e K_e = num_tokens * top_k``. The moe_sort
    routing kernel writes exactly that many entries to the
    ``tile_idx_to_expert_idx`` and ``tile_idx_to_mn_limit`` buffers (see
    ``include/flashinfer/trtllm/fused_moe/RoutingKernel.cuh``,
    ``ExclusiveSum`` of ``ceil(count[e] / tile_size)``). An under-tight
    bound would buffer-overflow at runtime; an over-loose bound wastes
    memory and slows autotune profiling.
    """

    @staticmethod
    def _trtllm_compact_formula(
        num_tokens: int, top_k: int, num_local_experts: int, tile_size: int
    ) -> int:
        """Reference implementation matching TRT-LLM's compact form
        in ``tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py``."""
        num_expanded_tokens = num_tokens * top_k
        if num_expanded_tokens <= num_local_experts:
            return num_expanded_tokens
        return (num_expanded_tokens + (tile_size - 1) * num_local_experts) // tile_size

    @staticmethod
    def _worst_case_actual_tile_count(
        num_tokens: int, top_k: int, num_local_experts: int, tile_size: int
    ) -> int:
        """Construct the worst-case routing distribution (one expert
        receives ``E - L + 1`` tokens, the others each receive 1 token)
        and compute the actual ``sum_e ceil(K_e / tile_size)``. This is
        the value the moe_sort kernel writes to ``num_non_exiting_tiles``
        in the worst case."""
        num_expanded_tokens = num_tokens * top_k
        L = num_local_experts
        T = tile_size
        if num_expanded_tokens <= L:
            return num_expanded_tokens
        per_expert = [1] * (L - 1) + [num_expanded_tokens - (L - 1)]
        assert sum(per_expert) == num_expanded_tokens
        return sum((k + T - 1) // T for k in per_expert)

    # DeepSeek-V3-style production shapes: num_experts=256, top_k=8,
    # num_tokens=16384 (prefill) — full EP × tile_size matrix.
    # EP=1 → num_local_experts=256, EP=8 → 32, EP=16 → 16, EP=32 → 8.
    _DEEPSEEK_V3_SHAPES = [
        pytest.param(16384, 8, 256, 128, id="dsv3-ep1-tile128"),
        pytest.param(16384, 8, 256, 256, id="dsv3-ep1-tile256"),
        pytest.param(16384, 8, 32, 128, id="dsv3-ep8-tile128"),
        pytest.param(16384, 8, 32, 256, id="dsv3-ep8-tile256"),
        pytest.param(16384, 8, 16, 128, id="dsv3-ep16-tile128"),
        pytest.param(16384, 8, 16, 256, id="dsv3-ep16-tile256"),
        pytest.param(16384, 8, 8, 128, id="dsv3-ep32-tile128"),
        pytest.param(16384, 8, 8, 256, id="dsv3-ep32-tile256"),
    ]

    # Generic MoE shapes covering diverse model configurations, so the
    # formula is exercised across model families (not only DeepSeek-V3).
    # Each entry is shaped (num_tokens, top_k, num_local_experts, tile_size).
    _GENERIC_MOE_SHAPES = [
        # Small MoEs: (num_experts=8, top_k=2)-style — Mixtral-8x7B family.
        pytest.param(4096, 2, 8, 128, id="generic-mixtral-style-ep1-tile128"),
        pytest.param(4096, 2, 8, 256, id="generic-mixtral-style-ep1-tile256"),
        # Mid-size MoEs: (num_experts=16, top_k=4)-style — DBRX family.
        pytest.param(4096, 4, 16, 128, id="generic-dbrx-style-ep1-tile128"),
        pytest.param(4096, 4, 16, 256, id="generic-dbrx-style-ep1-tile256"),
        pytest.param(2048, 4, 4, 128, id="generic-dbrx-style-ep4-tile128"),
        # Mid-large MoEs: (num_experts=64, top_k=6)-style.
        pytest.param(8192, 6, 64, 128, id="generic-64expert-ep1-tile128"),
        pytest.param(8192, 6, 8, 128, id="generic-64expert-ep8-tile128"),
        pytest.param(8192, 6, 8, 256, id="generic-64expert-ep8-tile256"),
        # Large MoEs with high top_k beyond DeepSeek-V3's 8 — guards
        # against future top_k regimes.
        pytest.param(4096, 16, 32, 128, id="generic-topk16-ep4-tile128"),
        pytest.param(4096, 16, 32, 256, id="generic-topk16-ep4-tile256"),
    ]

    # Edge cases: boundary values and divisibility cases that have
    # historically tripped off-by-one logic. Independent of any specific
    # model.
    _EDGE_CASE_SHAPES = [
        # num_expanded == num_local_experts (early-return boundary).
        pytest.param(4, 8, 32, 128, id="edge-expanded-equals-local"),
        # num_expanded just past num_local_experts — first non-trivial
        # branch path.
        pytest.param(5, 8, 32, 128, id="edge-expanded-just-past-local"),
        # (E - L) exactly divisible by tile_size — historically the
        # off-by-one case.
        pytest.param(20, 8, 32, 128, id="edge-remainder-zero"),
        # (E - L) one short of divisibility.
        pytest.param(19, 8, 32, 128, id="edge-remainder-near-zero"),
        # Smallest meaningful: num_tokens=1.
        pytest.param(1, 8, 32, 128, id="edge-num-tokens-1"),
        # Smallest meaningful: top_k=1.
        pytest.param(128, 1, 8, 128, id="edge-top-k-1"),
    ]

    @pytest.mark.parametrize(
        "num_tokens,top_k,num_local_experts,tile_size",
        _DEEPSEEK_V3_SHAPES + _GENERIC_MOE_SHAPES + _EDGE_CASE_SHAPES,
    )
    def test_matches_trtllm_compact_formula(
        self,
        num_tokens: int,
        top_k: int,
        num_local_experts: int,
        tile_size: int,
    ) -> None:
        """Pin ``get_max_num_tiles`` to TRT-LLM's compact closed-form
        across diverse MoE shape configurations.

        Coverage:
        - DeepSeek-V3 production shapes at every supported EP partition
          (EP=1, 8, 16, 32) × every tile_size (128, 256).
        - Generic MoE shapes representing other model families (small,
          medium, large; varied top_k and EP partitions).
        - Edge cases (boundary values, divisibility cases).
        """
        from flashinfer.fused_moe.cute_dsl.moe_utils import get_max_num_tiles

        actual = get_max_num_tiles(num_tokens, top_k, num_local_experts, tile_size)
        expected = self._trtllm_compact_formula(
            num_tokens, top_k, num_local_experts, tile_size
        )
        assert actual == expected, (
            f"get_max_num_tiles({num_tokens=}, {top_k=}, {num_local_experts=}, "
            f"{tile_size=}) returned {actual}; TRT-LLM compact formula "
            f"returns {expected}. A discrepancy may indicate drift away "
            f"from the upstream formula or an off-by-one in floor/ceil division."
        )

    @pytest.mark.parametrize(
        "num_tokens,top_k,num_local_experts,tile_size",
        _DEEPSEEK_V3_SHAPES + _GENERIC_MOE_SHAPES + _EDGE_CASE_SHAPES,
    )
    def test_worst_case_construction_is_tight(
        self,
        num_tokens: int,
        top_k: int,
        num_local_experts: int,
        tile_size: int,
    ) -> None:
        """The bound must equal the actual tile count produced by the
        worst-case routing distribution. If ``get_max_num_tiles`` exceeds
        the worst case, buffers are over-allocated; if it falls short,
        runtime will buffer-overflow."""
        from flashinfer.fused_moe.cute_dsl.moe_utils import get_max_num_tiles

        bound = get_max_num_tiles(num_tokens, top_k, num_local_experts, tile_size)
        worst_case = self._worst_case_actual_tile_count(
            num_tokens, top_k, num_local_experts, tile_size
        )
        assert bound == worst_case, (
            f"get_max_num_tiles({num_tokens=}, {top_k=}, {num_local_experts=}, "
            f"{tile_size=}) returned {bound}; worst-case routing "
            f"(one expert gets E - L + 1 tokens, others get 1) actually "
            f"produces {worst_case}. Mismatch implies the formula is "
            f"either over- or under-allocating."
        )

    def test_zero_tokens(self) -> None:
        """Zero tokens => zero tiles."""
        from flashinfer.fused_moe.cute_dsl.moe_utils import get_max_num_tiles

        assert get_max_num_tiles(0, 8, 32, 128) == 0
        assert get_max_num_tiles(0, 8, 32, 256) == 0

    def test_below_or_at_local_experts_threshold(self) -> None:
        """When ``num_expanded <= num_local_experts``, every token can
        go to a distinct expert and each contributes 1 fully-padded
        tile. Output equals num_expanded."""
        from flashinfer.fused_moe.cute_dsl.moe_utils import get_max_num_tiles

        assert get_max_num_tiles(2, 8, 32, 128) == 16  # E = 16 < L = 32
        assert get_max_num_tiles(4, 8, 32, 128) == 32  # E = 32 == L
        assert get_max_num_tiles(2, 8, 32, 256) == 16  # tile_size irrelevant here

    @pytest.mark.parametrize(
        "top_k,num_local_experts,tile_size",
        [
            # DeepSeek-V3 (num_experts=256, top_k=8) at every EP partition.
            pytest.param(8, 256, 128, id="dsv3-ep1-tile128"),
            pytest.param(8, 256, 256, id="dsv3-ep1-tile256"),
            pytest.param(8, 32, 128, id="dsv3-ep8-tile128"),
            pytest.param(8, 32, 256, id="dsv3-ep8-tile256"),
            pytest.param(8, 16, 128, id="dsv3-ep16-tile128"),
            pytest.param(8, 16, 256, id="dsv3-ep16-tile256"),
            pytest.param(8, 8, 128, id="dsv3-ep32-tile128"),
            pytest.param(8, 8, 256, id="dsv3-ep32-tile256"),
            # Generic shapes representing other model families.
            pytest.param(2, 8, 128, id="generic-mixtral-style"),
            pytest.param(4, 16, 128, id="generic-dbrx-style"),
            pytest.param(16, 32, 256, id="generic-topk16"),
        ],
    )
    def test_monotonic_in_num_tokens(
        self, top_k: int, num_local_experts: int, tile_size: int
    ) -> None:
        """Increasing ``num_tokens`` must never decrease the tile count,
        across the same DeepSeek-V3 EP × tile_size matrix and the
        same generic-model shapes covered by the formula tests above.
        Catches sign errors / off-by-one in future refactors."""
        from flashinfer.fused_moe.cute_dsl.moe_utils import get_max_num_tiles

        prev = -1
        for n in [1, 16, 256, 1024, 16384]:
            cur = get_max_num_tiles(n, top_k, num_local_experts, tile_size)
            assert cur >= prev, (
                f"non-monotonic at num_tokens={n}, top_k={top_k}, "
                f"num_local_experts={num_local_experts}, "
                f"tile_size={tile_size}: prev={prev}, cur={cur}"
            )
            prev = cur


@cute_dsl_available
class TestGetMaxNumPermutedTokens:
    """Tests for ``get_max_num_permuted_tokens``, which is defined as
    ``get_max_num_tiles * tile_size``."""

    @pytest.mark.parametrize(
        "num_tokens,top_k,num_local_experts,tile_size",
        TestGetMaxNumTiles._DEEPSEEK_V3_SHAPES
        + TestGetMaxNumTiles._GENERIC_MOE_SHAPES
        + TestGetMaxNumTiles._EDGE_CASE_SHAPES,
    )
    def test_consistent_with_get_max_num_tiles(
        self,
        num_tokens: int,
        top_k: int,
        num_local_experts: int,
        tile_size: int,
    ) -> None:
        """Result must equal ``get_max_num_tiles * tile_size`` exactly,
        across the same DeepSeek-V3, generic-model, and edge-case shape
        coverage as :class:`TestGetMaxNumTiles`."""
        from flashinfer.fused_moe.cute_dsl.moe_utils import (
            get_max_num_permuted_tokens,
            get_max_num_tiles,
        )

        permuted = get_max_num_permuted_tokens(
            num_tokens, top_k, num_local_experts, tile_size
        )
        tiles = get_max_num_tiles(num_tokens, top_k, num_local_experts, tile_size)
        assert permuted == tiles * tile_size


# =============================================================================
# Test Class: Autotuner bucket configuration (no GPU required)
# =============================================================================


@pytest.fixture(scope="module")
def bucket_spec():
    """The first ``DynamicTensorSpec`` of a default-configured
    ``CuteDslFusedMoENvfp4Runner`` — the spec that owns the
    ``gen_tuning_buckets`` / ``map_to_tuning_buckets`` callables under
    test. Module-scoped: the runner is stateless for these checks.
    """
    from flashinfer.fused_moe.cute_dsl.tuner import (
        CuteDslFusedMoENvfp4Runner,
    )

    runner = CuteDslFusedMoENvfp4Runner(
        forward_impl=lambda *a, **k: None,
        num_experts=256,
        top_k=8,
        num_local_experts=256,
    )
    return runner.tuning_config.dynamic_tensor_specs[0]


@cute_dsl_available
class TestAutotunerBucketConfig:
    """Structural tests for the ``gen_tuning_buckets`` /
    ``map_to_tuning_buckets`` configuration on
    ``CuteDslFusedMoENvfp4Runner.tuning_config``.

    These tests run without a GPU. They guard against bucket-config
    forms that bake a hardcoded cap into the autotuner's input-dim
    bucket logic. A capped form silently clamps the autotune to a
    fixed shape — at runtime any token count larger than the cap
    maps to the smaller cached bucket and uses a tactic profiled at
    the wrong workload size.

    The correct form passes the bucket generators as bare callables;
    the autotuner invokes them with the actual input dim at autotune
    time so the bucket set adapts to the workload.
    """

    def test_gen_tuning_buckets_is_callable_not_static_tuple(self, bucket_spec):
        """``gen_tuning_buckets`` must be a callable that adapts to the
        actual input dim at autotune time — not a pre-computed
        tuple/sequence that bakes in a hardcoded cap.
        """
        assert callable(bucket_spec.gen_tuning_buckets), (
            f"gen_tuning_buckets must be a callable that adapts to the "
            f"runtime input dim — got "
            f"{type(bucket_spec.gen_tuning_buckets).__name__}. A "
            f"pre-computed sequence (e.g., a tuple) likely indicates a "
            f"bucket set with a hardcoded cap; pass the bare function "
            f"reference instead."
        )

    def test_gen_tuning_buckets_responds_to_input_dim(self, bucket_spec):
        """Calling ``gen_tuning_buckets`` with successively larger input
        dims must produce bucket sets whose maximum grows with the
        input. A capped form would produce identical (capped) bucket
        sets regardless of input.
        """
        small = bucket_spec.gen_tuning_buckets(8192)
        medium = bucket_spec.gen_tuning_buckets(16384)
        large = bucket_spec.gen_tuning_buckets(32768)
        assert max(small) >= 8192, (
            f"gen_tuning_buckets(8192) max should reach 8192; got {max(small)}"
        )
        assert max(medium) >= 16384, (
            f"gen_tuning_buckets(16384) max should reach 16384; "
            f"got {max(medium)}. Likely a hardcoded cap below 16384."
        )
        assert max(large) >= 32768, (
            f"gen_tuning_buckets(32768) max should reach 32768; "
            f"got {max(large)}. Likely a hardcoded cap below 32768."
        )
        assert max(medium) > max(small), (
            f"larger input dim should produce larger bucket max, but "
            f"max(buckets@8192)={max(small)} >= "
            f"max(buckets@16384)={max(medium)} — likely a hardcoded cap."
        )

    @pytest.mark.parametrize("x", [16384, 32768, 65536])
    def test_map_to_tuning_buckets_responds_to_large_input(self, bucket_spec, x: int):
        """``map_to_tuning_buckets(x)`` for large x must return a value
        that scales with x, not collapse to a smaller constant. A
        capped form would silently return the cap value for any input
        above it.
        """
        result = bucket_spec.map_to_tuning_buckets(x)
        assert result >= x, (
            f"map_to_tuning_buckets({x}) = {result}; expected >= {x}. "
            f"Likely a hardcoded cap below {x}."
        )

    @pytest.mark.parametrize("x", [1, 2, 4, 8, 16, 32, 64, 128, 256])
    def test_map_to_tuning_buckets_matches_trtllm_at_small_powers_of_2(
        self, bucket_spec, x: int
    ):
        """At power-of-2 inputs in the small-N regime (≤ 256), fi's
        ``map_to_tuning_buckets(x)`` must equal ``x`` — matching
        TRT-LLM's ``last_positive_power_of_2(x)`` behavior. Locks in
        the fi/trt-llm parity that IS achievable in this regime.
        """
        from flashinfer.fused_moe.utils import last_positive_power_of_2

        result = bucket_spec.map_to_tuning_buckets(x)
        assert result == x == last_positive_power_of_2(x), (
            f"At x={x} (power of 2 in small-N regime), fi's "
            f"map_to_tuning_buckets should equal x and equal "
            f"last_positive_power_of_2(x) (TRT-LLM's pattern). Got "
            f"fi={result}, expected {x}."
        )

    def test_map_to_tuning_buckets_is_monotonic(self, bucket_spec):
        """``map_to_tuning_buckets`` must be monotonically non-decreasing
        in its input — a property TRT-LLM's mapper also satisfies.
        Catches a regression that would introduce non-monotonic
        bucket-mapping behavior.
        """
        test_xs = [
            1,
            2,
            8,
            100,
            256,
            257,
            512,
            768,
            1024,
            2048,
            2049,
            4096,
            4097,
            8192,
            16384,
            32768,
            65536,
        ]
        results = [bucket_spec.map_to_tuning_buckets(x) for x in test_xs]
        for prev_x, prev_y, curr_x, curr_y in zip(
            test_xs, results, test_xs[1:], results[1:], strict=False
        ):
            assert prev_y <= curr_y, (
                f"map_to_tuning_buckets must be monotonically "
                f"non-decreasing; got map({prev_x})={prev_y} > "
                f"map({curr_x})={curr_y}. Full mapping at probe "
                f"points: {list(zip(test_xs, results, strict=False))}."
            )

    @pytest.mark.parametrize("max_n", [256, 4096, 16384])
    def test_gen_tuning_buckets_covers_trtllm_power_of_2_points(
        self, bucket_spec, max_n: int
    ):
        """fi's bucket set must be a superset of TRT-LLM's power-of-2
        bucket set at every input dim tested, so the autotuner has at
        least the same coarse-grained coverage as TRT-LLM at every
        power-of-2 boundary up to the input dim.
        """
        from flashinfer.fused_moe.utils import last_positive_power_of_2

        fi_buckets = set(bucket_spec.gen_tuning_buckets(max_n))
        # Mirror TRT-LLM's get_last_power_of_2_num_tokens_buckets:
        # powers of 2 from 1 up to last_positive_power_of_2(max_n).
        trtllm_top = last_positive_power_of_2(max_n)
        trtllm_buckets = {1 << i for i in range(trtllm_top.bit_length())}
        missing = trtllm_buckets - fi_buckets
        assert not missing, (
            f"At max_n={max_n}, fi's bucket set is missing power-of-2 "
            f"values that TRT-LLM's bucket set would include: "
            f"{sorted(missing)}. fi: {sorted(fi_buckets)}; "
            f"TRT-LLM: {sorted(trtllm_buckets)}."
        )


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


# =============================================================================
# Test Class: moe_sort buffer-init invariants (poisoning)
# =============================================================================


@cute_dsl_available
@sm100_required
class TestMoeSortBufferInitPoisoned:
    """Validate the invariant that the routing kernel writes every
    output entry that downstream code reads, by pre-poisoning the
    wrapper's preallocated ``moe_sort`` output buffers with a sentinel
    value before the first call.

    The ``moe_sort`` wrapper in ``moe_utils.py`` allocates its output
    buffers via ``torch.empty(...)`` and relies on the routing kernel
    (``runPostTopKPipeline`` in ``trtllm_fused_moe_routing_common.cu``)
    to write every entry that any downstream kernel reads — including
    writing ``-1`` to masked slots of ``expanded_idx_to_permuted_idx``
    at EP > 1, and writing valid permuted indices to all unmasked
    slots. If the kernel is ever changed to skip an entry that
    downstream consumes, the uninitialized memory (or here, the
    sentinel) will leak through and produce dramatic numerical
    divergence — NaN/Inf via OOB index reads, or ``atomic-add`` into
    wildly wrong output rows.

    EP=32 specifically stresses the masked-position case for
    ``expanded_idx_to_permuted_idx``: with ``num_experts=256``,
    ``num_local_experts=8``, ``top_k=8`` only ~3.125% of expanded slots
    have their expert on this rank, so ~96% of the buffer must be
    written as ``-1`` by the kernel. If the kernel ever stops writing
    masked slots, this test catches it.

    These tests use ``use_cuda_graph=True`` so the wrapper preallocates
    buffers (the path where stale state from prior calls / poisoning is
    actually retained between calls). The default ``use_cuda_graph=False``
    path allocates fresh buffers per call and doesn't exercise the
    same scenario.
    """

    @pytest.mark.parametrize(
        "ep_size,num_tokens",
        [
            (1, 256),
            (8, 256),
            (16, 256),
            (32, 256),
            # High-N case exercises the `num_tokens > 1024` branch in
            # ``moe_sort`` where ``expert_counts`` is allocated per-call
            # via ``torch.empty`` and the vendored kernel relies on
            # ``launchInitExpertCounts`` to zero it before reading. No
            # other test in this suite exercises that branch — without
            # this case a future regression in the kernel-side init
            # would slip past CI.
            (8, 1280),
        ],
    )
    def test_wrapper_with_poisoned_moe_sort_buffers(
        self, ep_size: int, num_tokens: int
    ):
        """Pre-poison all six moe_sort output buffers with a sentinel
        before the wrapper's first call; verify output is well-formed
        and (at low N) matches the eager reference within tolerance.

        The high-N case (``num_tokens > 1024``) skips the
        ``compute_reference_moe_fp4`` comparison because that
        reference is ``O(num_tokens × top_k)`` Python iterations with
        ``.item()`` syncs and dominates CI runtime. The shape +
        NaN/Inf + non-zero-fraction assertions still run and are
        what the poisoning-detection logic actually relies on; the
        reference comparison is supplementary.
        """
        from flashinfer import CuteDslMoEWrapper

        hidden_size, intermediate_size = 256, 512
        num_experts, top_k = 256, 8
        num_local_experts = num_experts // ep_size
        local_expert_offset = 0  # rank 0; offset > 0 covered by TestExpertParallelism

        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            top_k=top_k,
        )

        # use_cuda_graph=True so the wrapper preallocates _moe_sort_buffers
        # (the path that retains stale state between calls — exactly what
        # we want to stress here).
        moe = CuteDslMoEWrapper(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_local_experts=num_local_experts,
            local_expert_offset=local_expert_offset,
            use_cuda_graph=True,
            max_num_tokens=num_tokens,
        )

        # Defensive guard: if a future refactor renames or restructures
        # ``_moe_sort_buffers``, the poisoning loop below would silently
        # iterate over zero items and the test would pass without
        # actually exercising the kernel-write invariant. Fail loudly
        # in that case so the test must be updated rather than silently
        # rotting.
        assert (
            getattr(moe, "_moe_sort_buffers", None) is not None
            and len(moe._moe_sort_buffers) > 0
        ), (
            "Wrapper no longer exposes a non-empty ``_moe_sort_buffers`` "
            "dict; the poisoning loop would be a no-op. Update this "
            "test to target the new preallocation attribute."
        )

        # Sentinel: a non-zero, non-(-1), out-of-valid-index-range int32.
        # If the kernel writes every entry that downstream reads, none of
        # these sentinels survive into gemm2 finalize. If any do, gemm2's
        # atomic-add will scatter into wildly wrong output rows, producing
        # NaN/Inf or massive numerical divergence.
        POISON = 0x7FFFFFFE
        for buf in moe._moe_sort_buffers.values():
            buf.fill_(POISON)

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
        assert not torch.isnan(result).any(), (
            f"ep_size={ep_size}, num_tokens={num_tokens}: NaNs in output "
            f"after running with poisoned moe_sort buffers — strong "
            f"indication that the routing kernel left sentinel values "
            f"in positions that downstream gemm2 finalize reads, causing "
            f"OOB / garbage-index atomic-adds. The invariant that the "
            f"kernel writes every consumed entry has been violated."
        )
        assert not torch.isinf(result).any(), (
            f"ep_size={ep_size}, num_tokens={num_tokens}: Infs in output "
            f"after running with poisoned moe_sort buffers — same failure "
            f"mode as NaN; kernel left sentinels in consumed positions."
        )

        # Sanity: verify SOMETHING non-trivial actually got computed.
        # At EP=32 with 256 global experts and top_k=8, ~22% of tokens
        # have at least one local expert; the rest produce all-zero
        # output. If the kernel were silently broken and returned all
        # zeros, this check would catch it.
        nonzero_fraction = (result.abs() > 1e-3).float().mean().item()
        # Lower bound proportional to local-expert coverage at this EP
        # (probability that at least one of top_k selected experts is on
        # this rank). EP=1 → ~100%; EP=8 → most tokens covered;
        # EP=16 → ~40%+; EP=32 → ~20%+. Bounds set well below the real
        # fraction to absorb FP4 quantization producing legitimately
        # small (sub-threshold) outputs.
        min_expected_nonzero = {1: 0.5, 8: 0.3, 16: 0.15, 32: 0.05}[ep_size]
        assert nonzero_fraction >= min_expected_nonzero, (
            f"ep_size={ep_size}, num_tokens={num_tokens}: only "
            f"{nonzero_fraction * 100:.2f}% of output entries are "
            f"non-zero (expected at least "
            f"{min_expected_nonzero * 100:.0f}%) — kernel may not have "
            f"executed correctly with poisoned buffers."
        )

        # Reference comparison is supplementary to the NaN/Inf +
        # non-zero-fraction checks above. Skip at high N because
        # ``compute_reference_moe_fp4`` is ``O(num_tokens × top_k)``
        # Python iterations with ``.item()`` syncs, dominating CI
        # runtime per case. The poisoning-detection signal lives in
        # the assertions already executed.
        if num_tokens > 1024:
            return

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
            f"ep_size={ep_size}, num_tokens={num_tokens}: poisoned-buffer "
            f"test failed: only {percent_within * 100:.2f}% within "
            f"tolerance (atol={atol:.4f}). The kernel left poison "
            f"sentinels in positions that downstream reads — the "
            f"invariant that the routing kernel writes every consumed "
            f"entry is violated, so a Python-side ``.fill_()`` / "
            f"``.zero_()`` init step is load-bearing at this EP."
        )


# =============================================================================
# Test Class: All Valid Tactics
# =============================================================================


@cute_dsl_available
@sm100_required
class TestAllValidTactics:
    """Test that every tactic returned by get_valid_tactics produces correct output.

    For each problem configuration, gets the filtered list of valid tactics via
    can_implement checks, then runs CuteDslMoEWrapper with each tactic explicitly
    and verifies numerical accuracy against the reference implementation.
    """

    @pytest.mark.parametrize(
        "num_tokens,hidden_size,intermediate_size,num_experts,top_k",
        [
            (128, 256, 512, 256, 2),
            (256, 1024, 2048, 256, 8),
        ],
    )
    def test_all_tactics_accuracy(
        self,
        num_tokens: int,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
    ):
        """Verify every valid tactic produces correct output."""
        from flashinfer import CuteDslMoEWrapper

        num_local_experts = num_experts

        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            top_k=top_k,
        )

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

        # Create wrapper without CUDA graph so we can freely try different tile_sizes
        moe = CuteDslMoEWrapper(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            use_cuda_graph=False,
        )

        # Get the filtered list of valid tactics for this problem size
        inputs = [
            tensors["x"],
            tensors["x_sf"],
            tensors["token_selected_experts"],
            tensors["token_final_scales"],
            tensors["w1_weight"],
            tensors["w1_weight_sf"],
            tensors["w1_alpha"],
            tensors["fc2_input_scale"],
            tensors["w2_weight"],
            tensors["w2_weight_sf"],
            tensors["w2_alpha"],
        ]
        valid_tactics = moe._runner.get_valid_tactics(inputs, None)
        assert len(valid_tactics) > 0, "No valid tactics found"

        num_passed = 0
        num_failed = 0
        for tactic in valid_tactics:
            tile_size = tactic[0]
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
                tactic=tactic,
            )

            assert result.shape == (num_tokens, hidden_size)
            assert not torch.isnan(result).any(), f"NaN in output for tactic {tactic}"
            assert not torch.isinf(result).any(), f"Inf in output for tactic {tactic}"

            passed, percent_within, atol = check_accuracy(result, ref_output)
            if passed:
                num_passed += 1
            else:
                num_failed += 1
                # Don't fail immediately; report all failures at the end
                print(
                    f"[FAIL] tactic tile_size={tile_size} "
                    f"gemm1={tactic[1][0]},{tactic[1][1]} "
                    f"gemm2={tactic[2][0]},{tactic[2][1]}: "
                    f"{percent_within * 100:.2f}% within tolerance (atol={atol:.4f})"
                )

        total = len(valid_tactics)
        assert num_failed == 0, (
            f"{num_failed}/{total} tactics failed accuracy check "
            f"(tokens={num_tokens}, hidden={hidden_size}, "
            f"intermediate={intermediate_size}, experts={num_experts}, top_k={top_k})"
        )


# =============================================================================
# Test Class: CuteDslMoEWrapper prealloc static invariants (no GPU required)
# =============================================================================


@cute_dsl_available
class TestPreallocStaticInvariants:
    """No-GPU structural invariants on ``VALID_TILE_SIZES``.

    The empirical buffer-shape and prealloc-gate behavior is covered
    by ``TestPreallocBuffersIntegration`` and
    ``TestPreallocGateUnderTuning`` (GPU-required).  This class catches
    the orthogonal failure mode where ``VALID_TILE_SIZES`` is
    accidentally reduced to a single entry — in that case the GPU
    integration tests pass trivially (no max/min divergence in
    ``_allocate_buffers``, only one tile_size to gate-check) and the
    bias-prevention silently disappears.
    """

    def test_valid_tile_sizes_has_multiple_entries(self):
        """``VALID_TILE_SIZES`` must enumerate more than one tile_size.
        With a single entry, the bias-prevention is moot — the
        autotuner only ever profiles one tile_size class, defeating
        the whole point of widening the prealloc.
        """
        from flashinfer.fused_moe.cute_dsl.tuner import VALID_TILE_SIZES

        assert len(VALID_TILE_SIZES) >= 2, (
            f"VALID_TILE_SIZES has only {len(VALID_TILE_SIZES)} entry; "
            f"need >= 2 for the prealloc-bias fix to be meaningful."
        )
        assert all(isinstance(t, int) and t > 0 for t in VALID_TILE_SIZES), (
            f"VALID_TILE_SIZES entries must be positive ints; got {VALID_TILE_SIZES}"
        )


# =============================================================================
# Test Class: CuteDslMoEWrapper prealloc-buffer integration (GPU required)
# =============================================================================


@cute_dsl_available
@sm100_required
class TestPreallocBuffersIntegration:
    """Verify the wrapper's prealloc'd buffers fit the workload at
    *every* ``tile_size in VALID_TILE_SIZES``, not just the
    constructor-time ``self.tile_size``.

    Load-bearing property: when the autotuner picks a tactic with
    ``tile_size != self.tile_size`` (the common case at large N where
    ``tile_size=256`` wins on intrinsic kernel time), the wrapper's
    ``use_prealloc`` gate still resolves True and inference uses the
    prealloc.  This requires the buffers to fit the *largest* possible
    workload across all valid tile_sizes; if they were sized only for
    ``self.tile_size``, picking a different tactic at runtime would
    OOB-write the prealloc -- forcing the gate to fall through to
    per-call ``torch.empty()`` calls, which violates the wrapper's
    CUDA-graph contract.
    """

    def test_prealloc_buffers_fit_all_valid_tile_sizes(self):
        from flashinfer import CuteDslMoEWrapper
        from flashinfer.fused_moe.cute_dsl.moe_utils import (
            get_max_num_permuted_tokens,
            get_max_num_tiles,
        )
        from flashinfer.fused_moe.cute_dsl.tuner import VALID_TILE_SIZES

        wrapper = CuteDslMoEWrapper(
            num_experts=256,
            top_k=8,
            hidden_size=256,
            intermediate_size=512,
            num_local_experts=256,
            local_expert_offset=0,
            use_cuda_graph=True,
            max_num_tokens=256,
        )

        gemm1_capacity = wrapper._gemm1_output.shape[0]
        gemm1_scale_capacity = wrapper._gemm1_output_scale.shape[0]
        permuted_idx_capacity = wrapper._moe_sort_buffers[
            "out_permuted_idx_to_expanded_idx"
        ].shape[0]
        tile_expert_capacity = wrapper._moe_sort_buffers[
            "out_tile_idx_to_expert_idx"
        ].shape[0]
        tile_mn_limit_capacity = wrapper._moe_sort_buffers[
            "out_tile_idx_to_mn_limit"
        ].shape[0]

        # Scale buffer is sized in scale-factor elements (one per
        # (permuted_token, scale_vec_group) pair), not in permuted
        # tokens directly.
        scale_factor_per_token = wrapper.intermediate_size // wrapper.sf_vec_size

        for tile_size in VALID_TILE_SIZES:
            required_permuted = get_max_num_permuted_tokens(
                wrapper.max_num_tokens,
                wrapper.top_k,
                wrapper.num_local_experts,
                tile_size,
            )
            required_scale_size = required_permuted * scale_factor_per_token
            required_tiles = get_max_num_tiles(
                wrapper.max_num_tokens,
                wrapper.top_k,
                wrapper.num_local_experts,
                tile_size,
            )

            assert gemm1_capacity >= required_permuted, (
                f"_gemm1_output rows ({gemm1_capacity}) < required "
                f"({required_permuted}) at tile_size={tile_size}"
            )
            assert gemm1_scale_capacity >= required_scale_size, (
                f"_gemm1_output_scale capacity ({gemm1_scale_capacity}) "
                f"< required ({required_scale_size} = {required_permuted} "
                f"permuted * {scale_factor_per_token} scales/token) at "
                f"tile_size={tile_size}"
            )
            assert permuted_idx_capacity >= required_permuted, (
                f"out_permuted_idx_to_expanded_idx capacity "
                f"({permuted_idx_capacity}) < required ({required_permuted}) "
                f"at tile_size={tile_size}"
            )
            assert tile_expert_capacity >= required_tiles, (
                f"out_tile_idx_to_expert_idx capacity "
                f"({tile_expert_capacity}) < required ({required_tiles}) "
                f"at tile_size={tile_size}"
            )
            assert tile_mn_limit_capacity >= required_tiles, (
                f"out_tile_idx_to_mn_limit capacity "
                f"({tile_mn_limit_capacity}) < required ({required_tiles}) "
                f"at tile_size={tile_size}"
            )


# =============================================================================
# Test Class: CuteDslMoEWrapper autotune-profiling prealloc gate (GPU required)
# =============================================================================


@cute_dsl_available
@sm100_required
class TestPreallocGateUnderTuning:
    """Validate that ``_forward_with_tactic``'s ``use_prealloc`` gate
    is on during normal inference (any valid tile_size) but off during
    the autotuner's per-tactic measurement window.

    Behavioral contract:

    1. **Inside the per-tactic measurement window** (i.e. while
       ``is_in_profile_measurement()`` is True): the gate must return
       ``False`` for every tactic, regardless of whether the tactic's
       ``tile_size`` matches ``self.tile_size``.  All tactics see the
       same per-call ``torch.empty()`` allocation overhead and the
       autotuner's tactic comparison is unbiased.

    2. **Inside ``autotune(True)`` but outside the measurement window**
       (cache lookups, ``do_preparation`` calls, the post-``choose_one``
       final invocation, concurrent threads): the gate must use
       prealloc for *any* ``tile_size in VALID_TILE_SIZES``.  This is
       the property that ``is_in_profile_measurement()`` adds over the
       broader ``is_tuning_mode`` flag: the gate doesn't leak into
       these adjacent code paths.

    3. **Outside any tuning context** (plain inference): same as case
       2 — prealloc for any ``tile_size in VALID_TILE_SIZES``.  This is
       the property that the expanded ``_allocate_buffers`` adds: the
       gate doesn't depend on ``tile_size == self.tile_size``, so
       whichever tactic the autotuner picks, the wrapper's CUDA-graph
       prealloc is still used and the wrapper's graph-safety contract
       is preserved.

    Implementation: monkey-patch the module-level ``_moe_core_impl``
    to capture the ``moe_sort_buffers`` argument without launching
    kernels, then call ``_forward_with_tactic`` from each of the three
    contexts × {``self.tile_size``, other valid tile_size}
    configurations.
    """

    def test_gate_decouples_self_tile_size_only_during_measurement_window(
        self, monkeypatch
    ):
        from flashinfer import CuteDslMoEWrapper, autotune
        from flashinfer.autotuner import _profile_measurement_scope
        from flashinfer.fused_moe.cute_dsl import fused_moe as fused_moe_module
        from flashinfer.fused_moe.cute_dsl.tuner import VALID_TILE_SIZES

        wrapper = CuteDslMoEWrapper(
            num_experts=256,
            top_k=8,
            hidden_size=256,
            intermediate_size=512,
            num_local_experts=256,
            local_expert_offset=0,
            use_cuda_graph=True,
            max_num_tokens=128,
        )

        # (context, tile_size) -> bool (prealloc'd buffers passed)
        captured: dict = {}
        # The mode under which the next call is made; updated by the
        # caller before each ``call(tile_size, mode)`` so the mock can
        # tag the captured row correctly.
        current_mode = {"name": "inference"}

        def mock_moe_core_impl(*args, **kwargs):
            captured[(current_mode["name"], kwargs["tile_size"])] = (
                kwargs["moe_sort_buffers"] is wrapper._moe_sort_buffers
            )
            n = args[0].shape[0] if args else kwargs["x"].shape[0]
            return torch.zeros(
                (n, wrapper.hidden_size), dtype=torch.bfloat16, device="cuda"
            )

        monkeypatch.setattr(fused_moe_module, "_moe_core_impl", mock_moe_core_impl)

        # Build minimal placeholder tensors: _forward_with_tactic only
        # reads x.shape[0]; everything else is passed through to the
        # (mocked) inner function untouched.
        n = 64  # < max_num_tokens=128 so the batch check passes
        x = torch.empty((n, wrapper.hidden_size // 2), dtype=torch.uint8, device="cuda")
        x_sf = torch.empty((n, 1), dtype=torch.uint8, device="cuda")
        token_selected_experts = torch.zeros(
            (n, wrapper.top_k), dtype=torch.int32, device="cuda"
        )
        token_final_scales = torch.zeros(
            (n, wrapper.top_k), dtype=torch.float32, device="cuda"
        )
        dummy_w = torch.empty((1,), dtype=torch.uint8, device="cuda")
        dummy_alpha = torch.empty((1,), dtype=torch.float32, device="cuda")

        def call(tile_size: int) -> None:
            wrapper._forward_with_tactic(
                x=x,
                x_sf=x_sf,
                token_selected_experts=token_selected_experts,
                token_final_scales=token_final_scales,
                w1_weight=dummy_w,
                w1_weight_sf=dummy_w,
                w1_alpha=dummy_alpha,
                fc2_input_scale=dummy_alpha,
                w2_weight=dummy_w,
                w2_weight_sf=dummy_w,
                w2_alpha=dummy_alpha,
                num_experts=wrapper.num_experts,
                top_k=wrapper.top_k,
                num_local_experts=wrapper.num_local_experts,
                tile_size=tile_size,
            )

        matching = wrapper.tile_size  # the tile_size the prealloc was sized for
        # Exercise every tile_size in VALID_TILE_SIZES so adding a new
        # entry doesn't silently leave the gate untested for that tile.
        others = [t for t in VALID_TILE_SIZES if t != matching]
        assert others, (
            f"Test requires >= 2 distinct VALID_TILE_SIZES entries; "
            f"got {VALID_TILE_SIZES}"
        )
        all_tiles = (matching, *others)

        # Context 1: inside autotune(True) AND inside the measurement
        # window — what _profile_single_kernel does for each tactic
        # invocation.  The gate must skip prealloc for every tactic.
        with autotune(True):
            with _profile_measurement_scope():
                current_mode["name"] = "measurement"
                for tile_size in all_tiles:
                    call(tile_size)

            # Context 2: inside autotune(True) but OUTSIDE the
            # measurement window — analogous to a cache hit, the
            # do_preparation call, or the runner invocation immediately
            # after choose_one returns. The gate should behave like
            # plain inference here.
            current_mode["name"] = "in_tuning_context_outside_measurement"
            for tile_size in all_tiles:
                call(tile_size)

        # Context 3: outside any tuning context — plain inference.
        current_mode["name"] = "inference"
        for tile_size in all_tiles:
            call(tile_size)

        # Context 1 contract: skip prealloc unconditionally.
        for tile_size in all_tiles:
            assert not captured[("measurement", tile_size)], (
                f"In the per-tactic measurement window, gate passed "
                f"prealloc'd buffers for tile_size={tile_size} "
                f"(self.tile_size={matching}). This re-introduces the "
                f"autotune-profiling bias the gate is designed to prevent."
            )

        # Context 2 contract: prealloc for ANY valid tile_size.  This
        # is the property that distinguishes
        # ``is_in_profile_measurement()`` from the broader
        # ``is_tuning_mode``: cache lookups, do_preparation calls,
        # post-choose_one runs, and concurrent threads should NOT lose
        # prealloc just because some other thread/operation is inside
        # an ``autotune(True)`` context.  Combined with the expanded
        # buffer sizing in ``_allocate_buffers``, the prealloc is also
        # used regardless of whether ``tile_size == self.tile_size``.
        for tile_size in all_tiles:
            assert captured[("in_tuning_context_outside_measurement", tile_size)], (
                f"Inside autotune(True) but outside the measurement "
                f"window at tile_size={tile_size}, gate did not pass "
                f"prealloc'd buffers (self.tile_size={matching}). "
                f"Either the narrower is_in_profile_measurement() "
                f"signal is leaking back into is_tuning_mode breadth, "
                f"or the gate is incorrectly checking tile_size == "
                f"self.tile_size -- both regress the wrapper's "
                f"CUDA-graph contract."
            )

        # Context 3 contract: same as Context 2 -- gate uses prealloc
        # for ANY valid tile_size.  This preserves the wrapper's
        # CUDA-graph contract regardless of which tactic the autotuner
        # picks at runtime.
        for tile_size in all_tiles:
            assert captured[("inference", tile_size)], (
                f"In inference mode at tile_size={tile_size}, gate "
                f"did not pass prealloc'd buffers "
                f"(self.tile_size={matching}). The wrapper loses its "
                f"CUDA-graph prealloc benefit -- with use_cuda_graph="
                f"True, captured graphs would record per-call "
                f"torch.empty() calls instead of using the prealloc, "
                f"violating the wrapper's run() graph-safety contract."
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
