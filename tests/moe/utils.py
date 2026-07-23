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
from contextlib import contextmanager
from enum import IntEnum

import pytest
import torch
from torch.nn import functional as F

from flashinfer import RoutingMethodType, is_gated_activation
from flashinfer.fused_moe import WeightLayout
from flashinfer.fused_moe.cute_dsl.moe_utils import (
    normalize_cute_dsl_moe_activation_type,
)
from flashinfer.tllm_enums import (
    ActivationType,
    DEFAULT_SWIGLU_ALPHA,
    DEFAULT_SWIGLU_BETA,
    DEFAULT_SWIGLU_LIMIT,
)
from flashinfer.utils import get_compute_capability


class QuantMode(IntEnum):
    """Supported quantization modes for MoE testing."""

    FP4_NVFP4_NVFP4 = 1
    FP4_MXFP4_MXFP8 = 2
    FP4_MXFP4_Bf16 = 3
    FP8_BLOCK_SCALE_DEEPSEEK = 4
    FP8_BLOCK_SCALE_MXFP8 = 5
    FP8_PER_TENSOR = 6
    BF16 = 7
    MXINT4_BF16_BF16 = 8


@contextmanager
def nvfp4_4over6_env(use_4over6: bool):
    original_value = os.environ.get("FLASHINFER_NVFP4_4OVER6", None)
    if use_4over6:
        os.environ["FLASHINFER_NVFP4_4OVER6"] = "1"
    else:
        os.environ["FLASHINFER_NVFP4_4OVER6"] = "0"

    try:
        yield
    finally:
        if original_value is None:
            os.environ.pop("FLASHINFER_NVFP4_4OVER6", None)
        else:
            os.environ["FLASHINFER_NVFP4_4OVER6"] = original_value


@pytest.fixture(autouse=True)
def set_nvfp4_4over6_env(request):
    if "use_4over6" not in request.fixturenames:
        yield
        return

    env_names = (
        "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH",
        "FLASHINFER_NVFP4_4OVER6",
        "FLASHINFER_NVFP4_4OVER6_ERR_MODE",
        "FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH",
        "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256",
    )
    original_values = {name: os.environ.get(name, None) for name in env_names}

    use_4over6 = request.getfixturevalue("use_4over6")
    os.environ["FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH"] = "1"
    os.environ["FLASHINFER_NVFP4_4OVER6"] = "1" if use_4over6 else "0"
    os.environ["FLASHINFER_NVFP4_4OVER6_ERR_MODE"] = "MAE"
    os.environ["FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH"] = "0"
    os.environ["FLASHINFER_NVFP4_4OVER6_E4M3_USE_256"] = "0"

    yield

    for name, value in original_values.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


NON_GATED_ACTIVATION_SUPPORTED_QUANT_MODES = [
    QuantMode.FP4_NVFP4_NVFP4,
    QuantMode.FP8_BLOCK_SCALE_MXFP8,
    QuantMode.FP8_PER_TENSOR,
    QuantMode.BF16,
]


def skip_checks(
    moe_impl,
    routing_config,
    weight_processing,
    activation_type,
    num_tokens,
    hidden_size,
    intermediate_size,
    logits_dtype,
    zero_hidden_states=False,
):
    """Common skip logic for all tests."""
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] not in [10]:
        pytest.skip("These tests are only guaranteed to work on SM100 and SM103 GPUs.")

    # Check moe_impl class by name to avoid circular imports
    is_fp4_moe = type(moe_impl).__name__ == "FP4Moe"
    is_fp8_block_scale_moe = type(moe_impl).__name__ == "FP8BlockScaleMoe"

    # Skip zero hidden states tests for non-FP8 Block Scale MoE implementations
    if zero_hidden_states and not is_fp8_block_scale_moe:
        pytest.skip("Skipping zero hidden states tests for non-FP8 Block Scale MoE.")

    # Skip incompatible combinations
    if activation_type == ActivationType.Geglu and (
        not is_fp4_moe
        or moe_impl.quant_mode != QuantMode.FP4_NVFP4_NVFP4
        or routing_config["routing_method_type"] != RoutingMethodType.TopK
        or num_tokens > 128
    ):
        pytest.skip(
            f"Incompatible: {moe_impl.name} + {activation_type} + {routing_config['routing_method_type']} + {num_tokens}"
        )
    elif activation_type == ActivationType.Swiglu and (
        hidden_size > 1024 or intermediate_size > 1024
    ):
        pytest.skip(
            f"Skip for testing speed: {activation_type} + {hidden_size} + {intermediate_size}"
        )

    compatible_activation_types = routing_config.get(
        "compatible_activation_types", None
    )
    if (
        compatible_activation_types is not None
        and activation_type not in compatible_activation_types
    ):
        pytest.skip(
            f"Incompatible: activation_type={activation_type} not in compatible_activation_types ({compatible_activation_types})"
        )

    if (
        not is_gated_activation(activation_type)
        and moe_impl.quant_mode not in NON_GATED_ACTIVATION_SUPPORTED_QUANT_MODES
    ):
        pytest.skip(
            f"Incompatible: {moe_impl.name} + {activation_type=} + quant_mode={moe_impl.quant_mode}: non-gated activations only supported with these quant modes: {NON_GATED_ACTIVATION_SUPPORTED_QUANT_MODES}"
        )

    # Skip large intermediate sizes for configurations with many experts
    if routing_config["num_experts"] > 512 and intermediate_size > 512:
        pytest.skip(
            f"Skipping for testing speed: intermediate_size={intermediate_size} with {routing_config['num_experts']} experts"
        )

    if type(moe_impl) not in routing_config["compatible_moe_impls"]:
        pytest.skip(
            f"Incompatible: {moe_impl.name} + {routing_config['routing_method_type'].name}"
        )
    if type(moe_impl) not in weight_processing["compatible_moe_impls"]:
        pytest.skip(
            f"Incompatible: {moe_impl.name} + {weight_processing['use_shuffled_weight']} + {weight_processing['layout']}"
        )
    if (
        is_fp8_block_scale_moe
        and moe_impl.fp8_quantization_type == QuantMode.FP8_BLOCK_SCALE_MXFP8
        and not weight_processing["use_shuffled_weight"]
    ):
        pytest.skip("use_shuffled_weight must be true for MxFp8.")
    if (
        is_fp8_block_scale_moe
        and moe_impl.fp8_quantization_type == QuantMode.FP8_BLOCK_SCALE_MXFP8
        and weight_processing["layout"] != WeightLayout.MajorK
    ):
        pytest.skip("weight_layout must be MajorK for MxFp8.")

    if intermediate_size not in routing_config["compatible_intermediate_size"]:
        pytest.skip(
            f"Incompatible: intermediate_size={intermediate_size} with {routing_config['routing_method_type'].name} routing ({routing_config['num_experts']} experts)"
        )

    if moe_impl.quant_mode == QuantMode.MXINT4_BF16_BF16 and (
        intermediate_size % 256 != 0 or hidden_size % 256 != 0
    ):
        pytest.skip(
            f"Incompatible: intermediate_size={intermediate_size} or hidden_size={hidden_size} with MXINT4_BF16_BF16 quantization"
        )

    # TODO(jimmzhou): enable MxFP4xBf16 on SM103
    if (
        is_fp4_moe
        and moe_impl.quant_mode == QuantMode.FP4_MXFP4_Bf16
        and compute_capability[0] == 10
        and compute_capability[1] == 3
    ):
        pytest.xfail(
            "Note(jimmzhou): Make MxFP4xBf16 nonfunctional on SM103 to avoid B200 regression"
        )

    if logits_dtype == torch.float32 and moe_impl.quant_mode not in [
        QuantMode.FP4_NVFP4_NVFP4,
        QuantMode.FP8_PER_TENSOR,
        QuantMode.FP8_BLOCK_SCALE_DEEPSEEK,
        QuantMode.FP8_BLOCK_SCALE_MXFP8,
        QuantMode.BF16,
    ]:
        pytest.skip(
            f"Incompatible: logits_dtype={logits_dtype} with {type(moe_impl).__name__} + {moe_impl.quant_mode}"
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


def quant_dequant_fp4_per_token_reference(
    tensor: torch.Tensor,
    global_scale_inv: torch.Tensor,
    sf_vec_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference per-token FP4 quantization for FC2 inputs.

    Returns the dequantized tensor before applying the row scale, plus the
    returned per-token scale. This mirrors the production MoE path where GEMM2
    consumes the quantized activation and the finalize kernel applies the row
    scale with the routing scale.
    """
    from flashinfer.quantization import (
        SfLayout,
        e2m1_and_ufp8sf_scale_to_float,
        nvfp4_quantize,
    )

    tensor_bf16 = tensor.to(torch.bfloat16)
    fp4_packed, sf, per_token_scale = nvfp4_quantize(
        tensor_bf16,
        global_scale_inv,
        sfLayout=SfLayout.layout_linear,
        sf_vec_size=sf_vec_size,
        backend="cuda",
        per_token_activation=True,
    )

    dequantized = e2m1_and_ufp8sf_scale_to_float(
        fp4_packed.cpu(),
        sf.view(torch.uint8).cpu(),
        torch.ones(1, dtype=torch.float32),
        sf_vec_size=sf_vec_size,
        ufp8_type=1,
        is_sf_swizzled_layout=False,
    ).to(tensor.device)

    return dequantized.float(), per_token_scale.to(tensor.device)


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
    fc2_input_scale: torch.Tensor | None = None,
    use_per_token_activation: bool = False,
    num_local_experts: int | None = None,
    local_expert_offset: int = 0,
    activation_type: int = ActivationType.Swiglu.value,
    activation: str | None = None,
    swiglu_alpha: float | None = None,
    swiglu_beta: float | None = None,
    swiglu_limit: float | None = None,
    gemm1_alpha: torch.Tensor | None = None,
    gemm2_alpha: torch.Tensor | None = None,
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
        use_per_token_activation: Use per-token activation.
        num_local_experts: Number of local experts (for EP). Defaults to num_experts.
        local_expert_offset: Starting expert ID for this EP rank. Defaults to 0.
        activation_type: GEMM1 activation type. Use ActivationType.Swiglu for
            gated SwiGLU/OAI and ActivationType.Relu2 for non-gated ReLU^2.
        activation: Optional B12x activation name. When provided, this takes
            precedence over activation_type.
        swiglu_alpha: SwiGLU sigmoid multiplier.
        swiglu_beta: SwiGLU up-projection bias.
        swiglu_limit: SwiGLU clamp limit.
        gemm1_alpha: GEMM1 per-expert scalar scales [num_local_experts]
        gemm2_alpha: GEMM2 per-expert scalar scales [num_local_experts]

    Returns:
        Output tensor [num_tokens, hidden_size]
    """
    if activation is None:
        _, gated = normalize_cute_dsl_moe_activation_type(activation_type)
        swiglu_alpha = DEFAULT_SWIGLU_ALPHA if swiglu_alpha is None else swiglu_alpha
        swiglu_beta = DEFAULT_SWIGLU_BETA if swiglu_beta is None else swiglu_beta
        swiglu_limit = DEFAULT_SWIGLU_LIMIT if swiglu_limit is None else swiglu_limit
    else:
        supported_activations = {
            "silu",
            "gelu_tanh",
            "swigluoai_uninterleave",
        }
        if activation not in supported_activations:
            raise ValueError(
                f"Unsupported B12x activation {activation!r}; "
                f"expected one of {sorted(supported_activations)}"
            )
        gated = True
        if activation == "swigluoai_uninterleave":
            swiglu_alpha = 1.702 if swiglu_alpha is None else swiglu_alpha
            swiglu_beta = 1.0 if swiglu_beta is None else swiglu_beta

    if num_local_experts is None:
        num_local_experts = num_experts

    device = hidden_states.device

    hidden_states = hidden_states.float()
    gemm1_weights = gemm1_weights.float()
    gemm2_weights = gemm2_weights.float()
    if gemm1_alpha is None:
        gemm1_alpha = torch.ones(num_local_experts, device=device, dtype=torch.float32)
    else:
        gemm1_alpha = gemm1_alpha.float()
    if gemm2_alpha is None:
        gemm2_alpha = torch.ones(num_local_experts, device=device, dtype=torch.float32)
    else:
        gemm2_alpha = gemm2_alpha.float()

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
            gemm1_out = gemm1_alpha[local_idx] * (token_input @ w1.T)

            per_token_scale = None
            if gated:
                linear = gemm1_out[:, :intermediate_size]
                gate = gemm1_out[:, intermediate_size:]
                if activation == "gelu_tanh":
                    act_out = F.gelu(gate, approximate="tanh") * linear
                elif activation == "silu":
                    act_out = silu(gate) * linear
                else:
                    if swiglu_limit is not None:
                        gate = gate.clamp(max=swiglu_limit)
                        linear = linear.clamp(min=-swiglu_limit, max=swiglu_limit)
                    act_out = (
                        gate
                        * torch.sigmoid(swiglu_alpha * gate)
                        * (linear + swiglu_beta)
                    )
            else:
                act_out = torch.relu(gemm1_out) ** 2

            if fc2_input_scale is not None:
                if use_per_token_activation:
                    act_out, per_token_scale = quant_dequant_fp4_per_token_reference(
                        act_out, fc2_input_scale, sf_vec_size=16
                    )
                else:
                    act_out = quant_dequant_fp4_reference(
                        act_out, fc2_input_scale, sf_vec_size=16
                    )

            w2 = gemm2_weights[local_idx]
            gemm2_out = act_out @ w2.T

            output_scale = scale * gemm2_alpha[local_idx]
            if per_token_scale is not None:
                output_scale = output_scale * per_token_scale[0]
            output[token_idx] += output_scale * gemm2_out.squeeze(0)

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
    gated: bool = True,
    use_per_token_activation: bool = False,
    interleave_gated_weights: bool = True,
    use_nontrivial_alphas: bool = True,
):
    """Create properly quantized MoE tensors for testing.

    CuTe SM100 kernels consume interleaved gated weights and the tests exercise
    nontrivial per-expert alpha scales. B12x kernels use the opposite settings;
    callers should use :func:`create_b12x_moe_tensors` for that configuration.
    """
    from flashinfer.fp4_quantization import fp4_quantize
    from flashinfer.quantization import (
        SfLayout,
        e2m1_and_ufp8sf_scale_to_float,
        nvfp4_quantize,
    )
    from flashinfer.quantization.nvfp4_quantization_utils import (
        current_nvfp4_4over6_config,
        make_nvfp4_global_scale,
    )
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

    torch.manual_seed(seed)
    sf_vec_size = 16

    # Input
    x_bf16 = (
        torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device) / 10
    )
    x_per_token_scale = None
    if use_per_token_activation:
        x_global_scale = make_nvfp4_global_scale(
            x_bf16,
            per_token_activation=True,
            nvfp4_4over6_config=current_nvfp4_4over6_config(),
        )
        x_quantized, x_sf, x_per_token_scale = nvfp4_quantize(
            x_bf16,
            x_global_scale,
            sfLayout=SfLayout.layout_linear,
            sf_vec_size=sf_vec_size,
            backend="cuda",
            per_token_activation=True,
        )
        x_ref = e2m1_and_ufp8sf_scale_to_float(
            x_quantized.cpu(),
            x_sf.view(torch.uint8).cpu().reshape(-1),
            torch.ones(1, dtype=torch.float32),
            sf_vec_size=sf_vec_size,
            ufp8_type=1,
            is_sf_swizzled_layout=False,
        ).to(device)
        x_ref = x_ref.float() * x_per_token_scale.unsqueeze(1)
    else:
        a1_gs = torch.tensor([1.0], device=device, dtype=torch.float32)
        x_quantized, x_sf = fp4_quantize(
            x_bf16,
            global_scale=a1_gs,
            sf_vec_size=sf_vec_size,
            is_sf_swizzled_layout=False,
        )
        x_ref = e2m1_and_ufp8sf_scale_to_float(
            x_quantized.cpu(),
            x_sf.view(torch.uint8).cpu().reshape(-1),
            torch.ones(1, dtype=torch.float32),
            sf_vec_size=sf_vec_size,
            ufp8_type=1,
            is_sf_swizzled_layout=False,
        ).to(device)
        x_ref = x_ref.float()
    x_sf = x_sf.unsqueeze(-1)

    # Routing
    router_logits = torch.randn(num_tokens, num_experts, device=device)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.float()
    selected_experts = selected_experts.to(torch.int32)

    # GEMM1 weights: gated SwiGLU has 2*intermediate rows (interleaved
    # linear+gate); non-gated ReLU^2 has a single intermediate-row projection.
    fc1_rows = 2 * intermediate_size if gated else intermediate_size
    w1_bf16 = (
        torch.randn(
            num_local_experts,
            fc1_rows,
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        / 10
    )

    w1_gs = torch.tensor([1.0], device=device, dtype=torch.float32)
    w1_for_quant = (
        interleave_linear_and_gate(w1_bf16, group_size=64, dim=1)
        if gated and interleave_gated_weights
        else w1_bf16
    )
    w1_flat = w1_for_quant.reshape(num_local_experts * fc1_rows, hidden_size)
    w1_q_flat, w1_sf_flat = fp4_quantize(
        w1_flat, global_scale=w1_gs, sf_vec_size=sf_vec_size, is_sf_swizzled_layout=True
    )
    w1_q = w1_q_flat.view(num_local_experts, fc1_rows, hidden_size // 2)
    w1_weight_sf = convert_sf_to_mma_layout(
        w1_sf_flat,
        m=fc1_rows,
        k=hidden_size,
        num_groups=num_local_experts,
        sf_vec_size=sf_vec_size,
    )
    if use_nontrivial_alphas:
        w1_alpha = torch.linspace(
            0.75, 1.25, num_local_experts, device=device, dtype=torch.float32
        )
    else:
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
    if use_nontrivial_alphas:
        w2_alpha = torch.linspace(
            1.25, 0.75, num_local_experts, device=device, dtype=torch.float32
        )
    else:
        w2_alpha = torch.ones(num_local_experts, device=device, dtype=torch.float32)

    fc2_input_scale = torch.tensor([1.0], device=device, dtype=torch.float32)

    return {
        "x": x_quantized,
        "x_sf": x_sf,
        "x_bf16": x_bf16,
        "x_ref": x_ref,
        "x_per_token_scale": x_per_token_scale,
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


def create_b12x_moe_tensors(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    num_local_experts: int,
    top_k: int,
    device: str = "cuda",
    seed: int = 42,
):
    """Create B12x MoE tensors with non-interleaved weights and unity alphas."""
    return create_moe_tensors(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        top_k=top_k,
        device=device,
        seed=seed,
        interleave_gated_weights=False,
        use_nontrivial_alphas=False,
    )


def create_relu2_moe_tensors(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    num_local_experts: int,
    top_k: int,
    device: str = "cuda",
    seed: int = 42,
):
    """Create MoE tensors for ReLU2 (non-gated: w1_rows = n, not 2*n)."""
    from flashinfer.fp4_quantization import fp4_quantize
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

    torch.manual_seed(seed)
    sf_vec_size = 16

    x_bf16 = (
        torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device) / 10
    )
    a1_gs = torch.tensor([1.0], device=device, dtype=torch.float32)
    x_quantized, x_sf = fp4_quantize(
        x_bf16,
        global_scale=a1_gs,
        sf_vec_size=sf_vec_size,
        is_sf_swizzled_layout=False,
    )
    x_sf = x_sf.unsqueeze(-1)

    router_logits = torch.randn(num_tokens, num_experts, device=device)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.float()
    selected_experts = selected_experts.to(torch.int32)

    # FC1 weights — NON-GATED: [E, n, k] instead of [E, 2*n, k]
    w1_bf16 = (
        torch.randn(
            num_local_experts,
            intermediate_size,
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        / 10
    )
    w1_gs = torch.tensor([1.0], device=device, dtype=torch.float32)
    w1_flat = w1_bf16.view(num_local_experts * intermediate_size, hidden_size)
    w1_q_flat, w1_sf_flat = fp4_quantize(
        w1_flat,
        global_scale=w1_gs,
        sf_vec_size=sf_vec_size,
        is_sf_swizzled_layout=True,
    )
    w1_q = w1_q_flat.view(num_local_experts, intermediate_size, hidden_size // 2)
    w1_weight_sf = convert_sf_to_mma_layout(
        w1_sf_flat,
        m=intermediate_size,
        k=hidden_size,
        num_groups=num_local_experts,
        sf_vec_size=sf_vec_size,
    )
    w1_alpha = torch.ones(num_local_experts, device=device, dtype=torch.float32)

    # FC2 weights — same shape as SiLU: [E, k, n]
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
        w2_flat,
        global_scale=w2_gs,
        sf_vec_size=sf_vec_size,
        is_sf_swizzled_layout=True,
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


def compute_reference_moe_relu2(
    hidden_states: torch.Tensor,
    fc1_weights: torch.Tensor,
    fc2_weights: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    fc2_input_scale: torch.Tensor | None,
) -> torch.Tensor:
    """Reference ReLU2 MoE: output = relu(FC1(x))^2, then FC2."""
    output = torch.zeros(num_tokens, hidden_size, dtype=torch.float32, device="cuda")

    for token_idx in range(num_tokens):
        token_input = hidden_states[token_idx].unsqueeze(0)

        for k in range(top_k):
            expert_idx = token_selected_experts[token_idx, k].item()
            scale = token_final_scales[token_idx, k].item()

            if expert_idx >= num_experts or expert_idx < 0:
                continue

            w1 = fc1_weights[expert_idx]
            fc1_out = token_input @ w1.T
            activated = torch.square(torch.relu(fc1_out))

            if fc2_input_scale is not None:
                activated = quant_dequant_fp4_reference(
                    activated, fc2_input_scale, sf_vec_size=16
                )

            w2 = fc2_weights[expert_idx]
            fc2_out = activated @ w2.T

            output[token_idx] += scale * fc2_out.squeeze(0)

    return output
