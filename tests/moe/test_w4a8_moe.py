"""Tests for W4A8 (INT4 x FP8_e4m3) MoE kernel on Hopper SM90.

Covers the mixed-dtype MoE GEMM path where:
  - weights are 4-bit signed integers (INT4, group-scaled),
  - activations are FP8 ``e4m3`` (per-tensor + per-channel pre-quant scales),
  - output is bfloat16 or float16.

Matches the semantics exercised by
``tests/moe/test_trtllm_cutlass_fused_moe.py::test_moe_w4a8`` with three test
preprocessing fixes (weight byte interleave, native-dtype act scales,
broadcast-max reference input scale; see ``_run_w4a8_moe`` below).

Shape envelope: the SM90 W4A8 kernel on its own carries enough FP8 + INT4
accumulation noise that ``torch.testing.assert_close(rtol=1e-2, atol=1e-1)``
vs a float32 PyTorch reference is only achievable at very small shapes
(``h == intermediate_size == 512``, ``num_experts == 2``). At larger hidden
size, intermediate size, or expert count the absolute error grows with K and
the number of accumulated expert contributions, and the strict tolerance is
exceeded. This matches the envelope used by the upstream CI test. Coverage
beyond that envelope is out of scope for this PR.
"""

from contextlib import nullcontext

import pytest
import torch
import torch.nn.functional as F

from flashinfer import fused_moe
from flashinfer.autotuner import autotune
from flashinfer.utils import is_sm90a_supported


# ============================================================================
# Helpers (subset copied from test_trtllm_cutlass_fused_moe.py for standalone use)
# ============================================================================


def _break_int4_bytes_to_int8(packed: torch.Tensor) -> torch.Tensor:
    """Unpack a uint8 tensor where each byte holds two signed int4 nibbles
    (low nibble first). Returns int8 with the last dim doubled."""
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    return torch.stack([low, high], dim=-1).reshape(packed.shape[0], -1)


def _dequantize_int4_to_dtype(
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    dtype: torch.dtype,
    weight_scale_2: torch.Tensor = None,
) -> torch.Tensor:
    unpacked = _break_int4_bytes_to_int8(packed_weight)
    scale_expanded = weight_scale.repeat_interleave(group_size, dim=1)
    dequant = unpacked.float() * scale_expanded.float()
    if weight_scale_2 is not None:
        dequant = dequant / weight_scale_2.float()
    return dequant.to(dtype)


def _compute_routing(router_logits, top_k):
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    return routing_weights.float(), selected_experts


def _torch_moe_w4a8(
    num_experts,
    x,
    w31_weight,
    w2_weight,
    selected_experts,
    routing_weights,
    fc1_input_scale,
    fc2_input_scale,
    fc1_pre_quant_scale,
    fc2_pre_quant_scale,
    fc1_weight_scale_2,
    fc2_weight_scale_2,
):
    dtype = x.dtype
    results = torch.zeros_like(x)
    for expert_id in range(num_experts):
        mask = selected_experts == expert_id
        if not mask.sum():
            continue
        batch_idx, nth_expert = torch.where(mask)
        w3_expert, w1_expert = torch.chunk(w31_weight[expert_id], 2, dim=0)
        w2_expert = w2_weight[expert_id]

        expert_inputs = x[batch_idx]
        scale1 = fc1_input_scale[expert_id]
        inp_scaled = expert_inputs * fc1_pre_quant_scale[expert_id]
        inp_q = (
            torch.clamp(inp_scaled / scale1, -448.0, 448.0)
            .to(torch.float8_e4m3fn)
            .to(dtype)
        )
        x1 = (inp_q @ w1_expert.t()) * scale1
        x2 = (inp_q @ w3_expert.t()) * scale1
        ws2 = fc1_weight_scale_2[expert_id]
        x1 = x1 * ws2.to(dtype)
        x2 = x2 * ws2.to(dtype)
        inter = F.silu(x1) * x2

        scale2 = fc2_input_scale[expert_id]
        inter_scaled = inter * fc2_pre_quant_scale[expert_id]
        inter_q = (
            torch.clamp(inter_scaled / scale2, -448.0, 448.0)
            .to(torch.float8_e4m3fn)
            .to(dtype)
        )
        output = (inter_q @ w2_expert.t()) * scale2
        ws2 = fc2_weight_scale_2[expert_id]
        output = output * ws2.to(dtype)

        results[batch_idx] += routing_weights[batch_idx, nth_expert, None] * output
    return results.view_as(x)


# ============================================================================
# Test configurations
# ============================================================================

# Strict-tolerance envelope. Going beyond h / intermediate_size == 512 or
# num_experts > 2 exceeds assert_close(rtol=1e-2, atol=1e-1) for the same
# reason the upstream CI test does; see module docstring.
CORRECTNESS_CONFIGS = [
    (4, 512, 2, 2, 512),
    (16, 512, 2, 2, 512),
]


# ============================================================================
# Core runner
# ============================================================================


def _run_w4a8_moe(
    batch_size: int,
    hidden_size: int,
    num_experts: int,
    top_k: int,
    intermediate_size: int,
    dtype: torch.dtype = torch.bfloat16,
    use_autotune: bool = False,
    strict_correctness: bool = True,
):
    torch.manual_seed(42)
    group_size = 128
    e = num_experts
    m = batch_size
    n = intermediate_size
    k = hidden_size
    affine_coeff = 0.005

    x = torch.randn(m, k, dtype=dtype, device="cuda")
    router_logits = torch.randn(m, e, dtype=dtype, device="cuda")

    # INT4 packed weights (two 4-bit signed ints per byte).
    w1_weight = torch.randint(0, 256, (e, n, k // 2), dtype=torch.uint8, device="cuda")
    w2_weight = torch.randint(0, 256, (e, k, n // 2), dtype=torch.uint8, device="cuda")
    w3_weight = torch.randint(0, 256, (e, n, k // 2), dtype=torch.uint8, device="cuda")

    # Per-group weight scales (one per group_size consecutive K elements).
    w1_scale = torch.randn(e, n, k // group_size, dtype=dtype, device="cuda") * affine_coeff
    w2_scale = torch.randn(e, k, n // group_size, dtype=dtype, device="cuda") * affine_coeff
    w3_scale = torch.randn(e, n, k // group_size, dtype=dtype, device="cuda") * affine_coeff

    # Per-channel pre-quant scales folded into FP8 activation quantization.
    w1_pre_quant_scale = torch.rand(e, k, dtype=dtype, device="cuda") * 0.1 + 0.95
    w2_pre_quant_scale = torch.rand(e, n, dtype=dtype, device="cuda") * 0.1 + 0.95
    w3_pre_quant_scale = torch.rand(e, k, dtype=dtype, device="cuda") * 0.1 + 0.95

    input_scale = torch.rand(e, 1, dtype=torch.float32, device="cuda") * 0.2 + 0.1
    weight_scale_2 = torch.ones(e, 1, dtype=torch.float32, device="cuda")

    # Concatenate gate/up weights along the N axis for the fused fc31 path.
    fc1_weights = torch.cat([w3_weight, w1_weight], dim=1)
    fc2_weights = w2_weight

    # Kernel-side weight byte interleave (ported from TRT-LLM PR #12451).
    fc1_weights_il = fused_moe.interleave_moe_weights_for_hopper_mixed_gemm(
        fc1_weights.contiguous().view(torch.uint8), "int4"
    )
    fc2_weights_il = fused_moe.interleave_moe_weights_for_hopper_mixed_gemm(
        fc2_weights.contiguous().view(torch.uint8), "int4"
    )

    def _interleave_scales(w: torch.Tensor, dim: int) -> torch.Tensor:
        """Reshape+permute scales matching TRT-LLM's WInt4AFP8 layout rules.

        factor = 4 if dim % 512 == 0 else (2 if dim % 256 == 0 else 1).
        """
        factor = 4 if dim % 512 == 0 else (2 if dim % 256 == 0 else 1)
        s = w.shape
        return (
            w.reshape(s[0], s[1], s[2] // factor, factor)
            .permute(0, 2, 1, 3)
            .reshape(s[0], s[2] // factor, s[1] * factor)
            .contiguous()
        )

    w3_w1_scales = torch.cat([w3_scale, w1_scale], dim=1)
    w3_w1_scales_int = _interleave_scales(w3_w1_scales, k)
    w2_scales_int = _interleave_scales(w2_scale, n)

    # Act scales stay in native dtype (consumed by expandInputRows / applyPrequantScale
    # as OutputType). Weight scales use the bf16 bit-pattern trick since the TMA load
    # path treats them as bf16 regardless of the module output dtype on SM90.
    w3_w1_pre_quant_max = torch.max(w1_pre_quant_scale, w3_pre_quant_scale)
    w3_w1_input_scale_max = input_scale.max()
    fc31_act_scale = (w3_w1_pre_quant_max / w3_w1_input_scale_max).to(dtype)
    fc2_act_scale = (w2_pre_quant_scale / input_scale).to(dtype).unsqueeze(-1)

    fc31_alpha = (weight_scale_2.squeeze(-1) * w3_w1_input_scale_max).float()
    fc2_alpha = (weight_scale_2.squeeze(-1) * input_scale.squeeze(-1)).float()

    zero_1 = torch.empty(0, dtype=dtype, device="cuda")
    zero_2 = torch.empty(0, dtype=dtype, device="cuda")

    w3_w1_scales_out = w3_w1_scales_int.to(torch.bfloat16).view(dtype)
    w2_scales_out = w2_scales_int.to(torch.bfloat16).view(dtype)

    quant_scales = (
        w3_w1_scales_out,
        w2_scales_out,
        fc31_act_scale,
        fc2_act_scale,
        zero_1,
        zero_2,
        fc31_alpha,
        fc2_alpha,
    )

    routing_weights, selected_experts = _compute_routing(router_logits, top_k)
    flash_output = torch.zeros_like(x)
    with autotune(True) if use_autotune else nullcontext():
        fused_moe.cutlass_fused_moe(
            x,
            selected_experts.to(torch.int32),
            routing_weights,
            fc1_weights_il,
            fc2_weights_il,
            dtype,
            quant_scales=quant_scales,
            use_w4_group_scaling=True,
            output=flash_output,
            use_packed_weights=True,
        )

    # Sanity: finite + correct shape.
    assert torch.isfinite(flash_output).all(), (
        f"Non-finite output: NaN={torch.isnan(flash_output).sum()}, "
        f"Inf={torch.isinf(flash_output).sum()}"
    )
    assert flash_output.shape == (m, k), (
        f"Shape mismatch: {flash_output.shape} vs ({m}, {k})"
    )

    # Reference: dequantize weights and run the MoE math in PyTorch.
    # We iterate over all experts (they're all weighted so there is no
    # active-subset optimization as in the MXFP4 path); kept simple and
    # independent of test_trtllm_cutlass_fused_moe.py.
    w31_weight_list = []
    w2_weight_list = []
    for e_idx in range(num_experts):
        w1_dq = _dequantize_int4_to_dtype(
            w1_weight[e_idx], w1_scale[e_idx], group_size, dtype, weight_scale_2[e_idx]
        )
        w3_dq = _dequantize_int4_to_dtype(
            w3_weight[e_idx], w3_scale[e_idx], group_size, dtype, weight_scale_2[e_idx]
        )
        w2_dq = _dequantize_int4_to_dtype(
            w2_weight[e_idx], w2_scale[e_idx], group_size, dtype, weight_scale_2[e_idx]
        )
        w31_weight_list.append(torch.cat([w3_dq, w1_dq], dim=0))
        w2_weight_list.append(w2_dq)
    w31_weight_dequant = torch.stack(w31_weight_list, dim=0)
    w2_weight_dequant = torch.stack(w2_weight_list, dim=0)

    # NB: fc1_input_scale for the reference is the broadcast max over experts
    # (the kernel folds per-expert input scales into a single divisor via
    # fc31_act_scale = pre_quant / input_scale_max). Using per-expert scales
    # here would double-correct and diverge.
    fc1_input_scale_for_ref = torch.full_like(
        input_scale.squeeze(-1), w3_w1_input_scale_max.item()
    )
    ref_output = _torch_moe_w4a8(
        num_experts,
        x,
        w31_weight_dequant,
        w2_weight_dequant,
        selected_experts,
        routing_weights,
        fc1_input_scale=fc1_input_scale_for_ref,
        fc2_input_scale=input_scale.squeeze(-1),
        fc1_pre_quant_scale=torch.max(w1_pre_quant_scale, w3_pre_quant_scale),
        fc2_pre_quant_scale=w2_pre_quant_scale,
        fc1_weight_scale_2=weight_scale_2.squeeze(-1),
        fc2_weight_scale_2=weight_scale_2.squeeze(-1),
    )

    if strict_correctness:
        torch.testing.assert_close(ref_output, flash_output, rtol=1e-2, atol=1e-1)
    else:
        diff = torch.abs(ref_output.float() - flash_output.float())
        tol = 0.1 + 1e-2 * torch.abs(ref_output.float())
        close_pct = (diff <= tol).float().mean().item()
        assert close_pct >= 0.999, (
            f"Only {close_pct:.4%} of elements within tolerance (need >= 99.9%). "
            f"max_abs_err={diff.max().item():.4f}"
        )

    return flash_output


# ============================================================================
# Tests
# ============================================================================


@pytest.mark.skipif(
    not is_sm90a_supported(torch.device("cuda")), reason="W4A8 MoE requires SM90"
)
@pytest.mark.parametrize(
    "batch_size,hidden_size,num_experts,top_k,intermediate_size",
    CORRECTNESS_CONFIGS,
    ids=[f"m{c[0]}_h{c[1]}_e{c[2]}_k{c[3]}" for c in CORRECTNESS_CONFIGS],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_w4a8_moe_correctness(
    batch_size, hidden_size, num_experts, top_k, intermediate_size, dtype
):
    """Strict correctness vs dequantized reference (small K, both output dtypes)."""
    _run_w4a8_moe(
        batch_size,
        hidden_size,
        num_experts,
        top_k,
        intermediate_size,
        dtype=dtype,
        use_autotune=False,
        strict_correctness=True,
    )


@pytest.mark.skipif(
    not is_sm90a_supported(torch.device("cuda")), reason="W4A8 MoE requires SM90"
)
def test_w4a8_moe_autotune():
    """Smoke test that autotune(True) doesn't break the W4A8 path."""
    _run_w4a8_moe(
        batch_size=16,
        hidden_size=512,
        num_experts=2,
        top_k=2,
        intermediate_size=512,
        dtype=torch.bfloat16,
        use_autotune=True,
        strict_correctness=True,
    )
