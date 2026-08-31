"""
Copyright (c) 2026 by FlashInfer team.

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

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from flashinfer.jit import env as jit_env

jit_env.FLASHINFER_AOT_DIR = Path("/tmp/no-flashinfer-aot")

import flashinfer.fused_moe as fused_moe
from flashinfer.utils import is_sm90a_supported

from .test_trtllm_cutlass_fused_moe import (
    _assert_close_with_error_stats,
    _compute_with_active_experts,
    _dequant_mxfp4_humming_prescale_on_device,
    _make_humming_e8m0_weight_scale,
    compute_routing,
)

pytestmark = pytest.mark.solo

_K_BLOCK = 32
_M = 16
_N = 512
_TOP_K = 2
_HUMMING_EPILOGUE_COMPENSATION = 64.0


@dataclass(frozen=True)
class AccuracyCase:
    name: str
    seed: int
    k: int
    num_experts: int
    distribution: str
    raw_scale: tuple[int, int] = (118, 122)
    assert_block_mean_lower: bool = False
    rtol: float = 5e-2
    atol: float = 1e-3
    max_bad: int = 0


ACCURACY_CASES = (
    AccuracyCase(
        name="gaussian-k512-e2",
        seed=3101,
        k=512,
        num_experts=2,
        distribution="gaussian",
    ),
    AccuracyCase(
        name="k32-heteroscedastic-k512-e2",
        seed=3105,
        k=512,
        num_experts=2,
        distribution="k32_heteroscedastic",
        assert_block_mean_lower=True,
    ),
    AccuracyCase(
        name="local-outlier-k512-e2",
        seed=3106,
        k=512,
        num_experts=2,
        distribution="local_outlier",
        assert_block_mean_lower=True,
    ),
    AccuracyCase(
        name="k32-heteroscedastic-k1024-e4",
        seed=3102,
        k=1024,
        num_experts=4,
        distribution="k32_heteroscedastic",
        assert_block_mean_lower=True,
    ),
    AccuracyCase(
        name="local-outlier-k2048-e8",
        seed=3103,
        k=2048,
        num_experts=8,
        distribution="local_outlier",
        assert_block_mean_lower=True,
    ),
    AccuracyCase(
        name="k32-heteroscedastic-k4096-e32",
        seed=3107,
        k=4096,
        num_experts=32,
        distribution="k32_heteroscedastic",
        assert_block_mean_lower=True,
    ),
    AccuracyCase(
        name="wide-weight-offset-k1024-e4",
        seed=3104,
        k=1024,
        num_experts=4,
        distribution="gaussian",
        raw_scale=(114, 128),
        rtol=2e-1,
        atol=5e-1,
        max_bad=2048,
    ),
)


def _make_activations(case: AccuracyCase, device: torch.device) -> torch.Tensor:
    x = torch.randn((_M, case.k), dtype=torch.float32, device=device) * 0.05
    if case.distribution == "gaussian":
        return x.to(torch.bfloat16)

    groups = x.view(_M, case.k // _K_BLOCK, _K_BLOCK)
    if case.distribution == "k32_heteroscedastic":
        # Adjacent K32 blocks deliberately span four orders of magnitude.
        pattern = torch.tensor(
            [1.0, 0.125, 0.015625, 0.001953125],
            dtype=torch.float32,
            device=device,
        )
        repeats = (groups.shape[1] + pattern.numel() - 1) // pattern.numel()
        scales = pattern.repeat(repeats)[: groups.shape[1]]
        groups.mul_(scales.view(1, -1, 1))
    elif case.distribution == "local_outlier":
        # A single spike must not determine the quantizer scale for unrelated blocks.
        groups.mul_(0.25)
        row_ids = torch.arange(_M, device=device)
        block_ids = row_ids.remainder(groups.shape[1])
        col_ids = (row_ids * 7).remainder(_K_BLOCK)
        signs = torch.where(row_ids.remainder(2) == 0, 2.0, -2.0)
        groups[row_ids, block_ids, col_ids] = signs
    else:
        raise ValueError(f"unknown activation distribution: {case.distribution}")
    return groups.view(_M, case.k).to(torch.bfloat16)


def _prepare_humming_weights(case: AccuracyCase, device: torch.device):
    e, k = case.num_experts, case.k
    w1 = torch.randint(0, 256, (e, 2 * _N, k // 2), dtype=torch.uint8, device=device)
    w2 = torch.randint(0, 256, (e, k, _N // 2), dtype=torch.uint8, device=device)
    low, high = case.raw_scale
    w1_raw_scale = _make_humming_e8m0_weight_scale(
        (e, 2 * _N, k // _K_BLOCK), device, low=low, high=high
    )
    w2_raw_scale = _make_humming_e8m0_weight_scale(
        (e, k, _N // _K_BLOCK), device, low=low, high=high
    )

    w1_processed, w1_offset, w1_residual = (
        fused_moe.preprocess_moe_weights_for_sm90_mixed_gemm_humming(
            w1, w1_raw_scale, interleave=False
        )
    )
    w2_processed, w2_offset, w2_residual = (
        fused_moe.preprocess_moe_weights_for_sm90_mixed_gemm_humming(
            w2, w2_raw_scale, interleave=False
        )
    )

    w1_interleaved = fused_moe.interleave_moe_weights_for_sm90_mixed_gemm(
        w1_processed, "fp4_fp8"
    )
    w2_interleaved = fused_moe.interleave_moe_weights_for_sm90_mixed_gemm(
        w2_processed, "fp4_fp8"
    )
    w1_scale_interleaved = fused_moe.interleave_moe_scales_for_sm90_mixed_gemm(
        w1_offset
    )
    w2_scale_interleaved = fused_moe.interleave_moe_scales_for_sm90_mixed_gemm(
        w2_offset
    )

    fc1_residual = (w1_residual * _HUMMING_EPILOGUE_COMPENSATION).contiguous()
    fc2_residual = (w2_residual * _HUMMING_EPILOGUE_COMPENSATION).contiguous()
    quant_scales = [
        w1_scale_interleaved.view(torch.int32),
        fc1_residual,
        torch.ones((), dtype=torch.float32, device=device),
        w2_scale_interleaved.view(torch.int32),
        fc2_residual,
    ]

    # The golden uses dequantized Humming weights, but never activation-quantizes
    # either GEMM input. Fold only the per-expert residuals into the FP32 weights.
    w1_dequant = _dequant_mxfp4_humming_prescale_on_device(
        w1_processed, w1_offset
    ).float()
    w2_dequant = _dequant_mxfp4_humming_prescale_on_device(
        w2_processed, w2_offset
    ).float()
    w31_by_expert = {
        expert_id: w1_dequant[expert_id] * fc1_residual[expert_id]
        for expert_id in range(e)
    }
    w2_by_expert = {
        expert_id: w2_dequant[expert_id] * fc2_residual[expert_id]
        for expert_id in range(e)
    }
    return (
        w1_interleaved,
        w2_interleaved,
        quant_scales,
        w31_by_expert,
        w2_by_expert,
    )


def _run_humming(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    quant_scales: list[torch.Tensor],
    *,
    use_act_block_scale: bool,
) -> torch.Tensor:
    output = torch.zeros_like(x)
    fused_moe.cutlass_fused_moe(
        x,
        selected_experts.to(torch.int32),
        routing_weights,
        w1,
        w2,
        torch.bfloat16,
        quant_scales=quant_scales,
        use_w4_group_scaling=True,
        use_wfp4afp8_humming=True,
        use_mxfp8_act_scaling=use_act_block_scale,
        output=output,
    )
    return output


def _error_stats(actual: torch.Tensor, golden: torch.Tensor) -> dict[str, float]:
    error = (actual.float() - golden.float()).abs().flatten()
    return {
        "mean": float(error.mean().item()),
        "max": float(error.max().item()),
        "p95": float(torch.quantile(error, 0.95).item()),
        "p99": float(torch.quantile(error, 0.99).item()),
    }


@pytest.mark.skipif(
    not is_sm90a_supported(torch.device("cuda")),
    reason="Humming per-token/per-block fused MoE accuracy requires SM90",
)
@pytest.mark.parametrize("case", ACCURACY_CASES, ids=lambda case: case.name)
def test_humming_per_token_vs_per_token_per_block_accuracy(case: AccuracyCase):
    torch.manual_seed(case.seed)
    device = torch.device("cuda")
    x = _make_activations(case, device)
    router_logits = torch.randn(
        (_M, case.num_experts), dtype=torch.bfloat16, device=device
    )
    routing_weights, selected_experts = compute_routing(router_logits, _TOP_K)

    w1, w2, quant_scales, w31_by_expert, w2_by_expert = _prepare_humming_weights(
        case, device
    )
    active_experts = torch.arange(case.num_experts, device=device)
    golden = _compute_with_active_experts(
        active_experts,
        x.float(),
        w31_by_expert,
        w2_by_expert,
        selected_experts,
        routing_weights.float(),
    )

    per_token = _run_humming(
        x,
        selected_experts,
        routing_weights,
        w1,
        w2,
        quant_scales,
        use_act_block_scale=False,
    )
    per_block = _run_humming(
        x,
        selected_experts,
        routing_weights,
        w1,
        w2,
        quant_scales,
        use_act_block_scale=True,
    )

    stats = {
        "per-token": _error_stats(per_token, golden),
        "per-token/per-block": _error_stats(per_block, golden),
    }
    for path, values in stats.items():
        print(
            f"{case.name}/{path}: mean={values['mean']:.6g} "
            f"max={values['max']:.6g} p95={values['p95']:.6g} "
            f"p99={values['p99']:.6g}"
        )
        actual = per_token if path == "per-token" else per_block
        _assert_close_with_error_stats(
            actual,
            golden,
            label=f"{case.name}/{path} vs shared unquantized FP32 golden",
            rtol=case.rtol,
            atol=case.atol,
            max_bad=case.max_bad,
        )

    if case.assert_block_mean_lower:
        assert stats["per-token/per-block"]["mean"] < stats["per-token"]["mean"], (
            f"{case.name}: expected K32 block scaling to reduce mean absolute error; "
            f"per-token={stats['per-token']['mean']:.6g}, "
            f"per-token/per-block={stats['per-token/per-block']['mean']:.6g}"
        )
