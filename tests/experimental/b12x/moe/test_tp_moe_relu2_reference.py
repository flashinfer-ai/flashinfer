from __future__ import annotations

import pytest
import torch

from b12x._lib.intrinsics import (
    FLOAT4_E2M1_MAX,
    fp4_quantize_values_torch,
    pack_grouped_fp4_values,
    swizzle_block_scale,
)
import b12x.moe.fused_moe._impl as tp_moe
from b12x.moe.fused_moe._impl import clear_tp_moe_caches
from b12x.moe._shared.kernels.reference import compare_to_reference, moe_reference_nvfp4

from b12x.testing.reference.helpers import (
    prepare_tp_moe_fp4_experts,
    require_b12x,
    run_tp_moe_fp4,
)


BACKEND_CASES = [
    ("micro", 2, 10_000),
    ("dynamic_mid", 128, 10_000),
    ("dynamic_large", 768, 0),
]


def _quantize_moe_weight_storage(
    input_tensor: torch.Tensor,
    global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_groups, rows, cols = input_tensor.shape
    quantized = torch.zeros(
        (num_groups, rows, cols), dtype=torch.float32, device=input_tensor.device
    )
    scales = torch.zeros(
        (num_groups, rows, cols // 16), dtype=torch.float32, device=input_tensor.device
    )
    for group_idx in range(num_groups):
        x = input_tensor[group_idx].float()
        sliced = x.view(rows, cols // 16, 16)
        block_max = sliced.abs().amax(dim=-1, keepdim=True)
        scale = (
            (global_scale[group_idx] * (block_max / FLOAT4_E2M1_MAX))
            .to(torch.float8_e4m3fn)
            .to(torch.float32)
        )
        output_scale = 1.0 / (scale * (1.0 / global_scale[group_idx]))
        clipped = torch.clamp(
            sliced * output_scale, -FLOAT4_E2M1_MAX, FLOAT4_E2M1_MAX
        ).view(rows, cols)
        quantized[group_idx] = fp4_quantize_values_torch(clipped)
        scales[group_idx] = scale.squeeze(-1)

    packed = pack_grouped_fp4_values(quantized).permute(2, 0, 1).contiguous()
    swizzled = swizzle_block_scale(scales.to(torch.float8_e4m3fn))
    return packed, swizzled


def _make_activation_case(
    *,
    device: torch.device,
    activation: str,
    m: int,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)

    E, k, n, topk = 1, 128, 128, 1
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    topk_ids = torch.zeros(m, topk, device=device, dtype=torch.int32)
    topk_weights = torch.ones(m, topk, device=device, dtype=torch.float32)

    w1_rows = 2 * n if activation in {"silu", "situ"} else n
    w1 = torch.randn(E, w1_rows, k, device=device, dtype=torch.bfloat16) * 0.5
    w2 = torch.randn(E, k, n, device=device, dtype=torch.bfloat16) * 0.25
    a1_gscale = torch.ones(E, device=device, dtype=torch.float32)
    a2_gscale = torch.ones(E, device=device, dtype=torch.float32)
    w1_fp4, w1_blockscale = _quantize_moe_weight_storage(w1, a1_gscale)
    w2_fp4, w2_blockscale = _quantize_moe_weight_storage(w2, a2_gscale)
    w1_alphas = torch.ones(E, device=device, dtype=torch.float32)
    w2_alphas = torch.ones(E, device=device, dtype=torch.float32)
    return (
        x,
        topk_ids,
        topk_weights,
        w1_fp4,
        w1_blockscale,
        w1_alphas,
        w2_fp4,
        w2_blockscale,
        w2_alphas,
        a1_gscale,
        a2_gscale,
        E,
        k,
        n,
    )


def _run_activation_case(
    *,
    activation: str,
    m: int,
    micro_dynamic_cutover: int,
    fast_math: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = require_b12x()
    (
        x,
        topk_ids,
        topk_weights,
        w1_fp4,
        w1_blockscale,
        w1_alphas,
        w2_fp4,
        w2_blockscale,
        w2_alphas,
        a1_gscale,
        a2_gscale,
        E,
        k,
        n,
    ) = _make_activation_case(device=device, activation=activation, m=m)

    reference = moe_reference_nvfp4(
        x,
        w1_fp4,
        w1_blockscale,
        w1_alphas,
        w2_fp4,
        w2_blockscale,
        w2_alphas,
        a1_gscale,
        a2_gscale,
        topk_ids,
        topk_weights,
        E,
        k,
        n,
        activation=activation,
    )
    experts = prepare_tp_moe_fp4_experts(
        a=x,
        a1_gscale=a1_gscale,
        w1_fp4=w1_fp4,
        w1_blockscale=w1_blockscale,
        w1_alphas=w1_alphas,
        a2_gscale=a2_gscale,
        w2_fp4=w2_fp4,
        w2_blockscale=w2_blockscale,
        w2_alphas=w2_alphas,
        activation=activation,
    )

    previous_cutover = dict(tp_moe._MICRO_DYNAMIC_CUTOVER_PAIRS_CACHE)
    try:
        clear_tp_moe_caches()
        tp_moe._MICRO_DYNAMIC_CUTOVER_PAIRS_CACHE["nvfp4"] = micro_dynamic_cutover

        output = run_tp_moe_fp4(
            a=x,
            experts=experts,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            input_scales_static=True,
            fast_math=fast_math,
        )
        torch.cuda.synchronize()
    finally:
        clear_tp_moe_caches()
        tp_moe._MICRO_DYNAMIC_CUTOVER_PAIRS_CACHE.update(previous_cutover)

    return output, reference


def _run_single_token_multi_expert_case(
    *,
    activation: str,
    topk_ids_dtype: torch.dtype,
    micro_dynamic_cutover: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = require_b12x()
    torch.manual_seed(7)

    m, E, k, n = 1, 4, 128, 128
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    topk_ids = torch.tensor([[3, 1, 2]], device=device, dtype=topk_ids_dtype)
    topk_logits = torch.tensor([[0.2, -0.1, 0.4]], device=device, dtype=torch.float32)
    topk_weights = torch.softmax(topk_logits, dim=-1)

    w1_rows = 2 * n if activation in {"silu", "situ"} else n
    w1 = torch.randn(E, w1_rows, k, device=device, dtype=torch.bfloat16) * 0.5
    w2 = torch.randn(E, k, n, device=device, dtype=torch.bfloat16) * 0.25
    a1_gscale = torch.ones(E, device=device, dtype=torch.float32)
    a2_gscale = torch.ones(E, device=device, dtype=torch.float32)
    w1_fp4, w1_blockscale = _quantize_moe_weight_storage(w1, a1_gscale)
    w2_fp4, w2_blockscale = _quantize_moe_weight_storage(w2, a2_gscale)
    w1_alphas = torch.ones(E, device=device, dtype=torch.float32)
    w2_alphas = torch.ones(E, device=device, dtype=torch.float32)

    reference = moe_reference_nvfp4(
        x,
        w1_fp4,
        w1_blockscale,
        w1_alphas,
        w2_fp4,
        w2_blockscale,
        w2_alphas,
        a1_gscale,
        a2_gscale,
        topk_ids,
        topk_weights,
        E,
        k,
        n,
        activation=activation,
    )
    experts = prepare_tp_moe_fp4_experts(
        a=x,
        a1_gscale=a1_gscale,
        w1_fp4=w1_fp4,
        w1_blockscale=w1_blockscale,
        w1_alphas=w1_alphas,
        a2_gscale=a2_gscale,
        w2_fp4=w2_fp4,
        w2_blockscale=w2_blockscale,
        w2_alphas=w2_alphas,
        activation=activation,
    )

    previous_cutover = dict(tp_moe._MICRO_DYNAMIC_CUTOVER_PAIRS_CACHE)
    try:
        clear_tp_moe_caches()
        tp_moe._MICRO_DYNAMIC_CUTOVER_PAIRS_CACHE["nvfp4"] = micro_dynamic_cutover

        output = run_tp_moe_fp4(
            a=x,
            experts=experts,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            input_scales_static=True,
            fast_math=False,
        )
        torch.cuda.synchronize()
    finally:
        clear_tp_moe_caches()
        tp_moe._MICRO_DYNAMIC_CUTOVER_PAIRS_CACHE.update(previous_cutover)

    return output, reference


@pytest.mark.parametrize(
    ("backend", "m", "micro_dynamic_cutover"),
    BACKEND_CASES,
)
@pytest.mark.parametrize("activation", ["silu", "relu2"])
def test_activation_exact_path_matches_reference_across_backends(
    activation: str,
    backend: str,
    m: int,
    micro_dynamic_cutover: int,
) -> None:
    output, reference = _run_activation_case(
        activation=activation,
        m=m,
        micro_dynamic_cutover=micro_dynamic_cutover,
        fast_math=False,
    )
    metrics = compare_to_reference(output, reference)
    assert metrics.max_abs == 0.0, f"{activation}/{backend}: {metrics}"
    assert metrics.rmse == 0.0, f"{activation}/{backend}: {metrics}"


@pytest.mark.parametrize(
    ("backend", "m", "micro_dynamic_cutover"),
    BACKEND_CASES,
)
def test_relu2_matches_reference_across_backends(
    backend: str,
    m: int,
    micro_dynamic_cutover: int,
) -> None:
    output, reference = _run_activation_case(
        activation="relu2",
        m=m,
        micro_dynamic_cutover=micro_dynamic_cutover,
        fast_math=True,
    )
    metrics = compare_to_reference(output, reference)
    assert metrics.max_abs == 0.0, f"{backend}: {metrics}"
    assert metrics.rmse == 0.0, f"{backend}: {metrics}"


@pytest.mark.parametrize("activation", ["silu", "relu2"])
def test_single_token_multi_expert_micro_matches_int32_with_int64_topk_ids(
    activation: str,
) -> None:
    output_i64, reference = _run_single_token_multi_expert_case(
        activation=activation,
        topk_ids_dtype=torch.int64,
        micro_dynamic_cutover=128,
    )
    output_i32, _ = _run_single_token_multi_expert_case(
        activation=activation,
        topk_ids_dtype=torch.int32,
        micro_dynamic_cutover=128,
    )
    pair_metrics = compare_to_reference(output_i64, output_i32)
    assert pair_metrics.cos > 0.9999, f"{activation} int64 vs int32: {pair_metrics}"

    metrics = compare_to_reference(output_i64, reference)
    assert metrics.cos > 0.9999, f"{activation}: {metrics}"


def test_single_token_multi_expert_situ_avoids_micro_and_matches_reference() -> None:
    implementation, _, _ = tp_moe._resolve_workspace_layout(
        num_tokens=1,
        weight_E=4,
        num_topk=3,
        k=128,
        n=128,
        activation="situ",
        quant_mode="nvfp4",
    )
    assert implementation == "dynamic"

    output_i64, reference = _run_single_token_multi_expert_case(
        activation="situ",
        topk_ids_dtype=torch.int64,
        micro_dynamic_cutover=128,
    )
    output_i32, _ = _run_single_token_multi_expert_case(
        activation="situ",
        topk_ids_dtype=torch.int32,
        micro_dynamic_cutover=128,
    )
    pair_metrics = compare_to_reference(output_i64, output_i32)
    assert pair_metrics.cos > 0.9999, f"situ int64 vs int32: {pair_metrics}"

    metrics = compare_to_reference(output_i64, reference)
    assert metrics.cos > 0.9999, f"situ: {metrics}"


@pytest.mark.parametrize("m", [1, 2, 4])
def test_silu_tiny_dynamic_direct_matches_reference(m: int) -> None:
    output, reference = _run_activation_case(
        activation="silu",
        m=m,
        micro_dynamic_cutover=0,
        fast_math=False,
    )
    metrics = compare_to_reference(output, reference)
    assert metrics.max_abs == 0.0, f"m={m}: {metrics}"
    assert metrics.rmse == 0.0, f"m={m}: {metrics}"


def test_silu_single_token_multi_expert_dynamic_direct_matches_reference() -> None:
    output, reference = _run_single_token_multi_expert_case(
        activation="silu",
        topk_ids_dtype=torch.int32,
        micro_dynamic_cutover=0,
    )
    metrics = compare_to_reference(output, reference)
    assert metrics.cos > 0.9999, f"silu/direct: {metrics}"


def test_dynamic_deterministic_multislice_matches_atomic_and_repeats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Grouped deterministic FC2 slices preserve FC1 input and accumulate."""
    device = require_b12x()
    torch.manual_seed(20260716)
    m, experts_count, k, n, topk = 29, 32, 256, 384, 6

    x = torch.randn(m, k, device=device, dtype=torch.bfloat16) * 0.2
    topk_logits, topk_ids = torch.topk(
        torch.randn(m, experts_count, device=device), topk, dim=-1
    )
    topk_ids = topk_ids.to(torch.int32)
    topk_weights = torch.softmax(topk_logits, dim=-1)
    w1_fp4 = torch.randint(
        0,
        256,
        (experts_count, 2 * n, k // 2),
        device=device,
        dtype=torch.uint8,
    )
    w2_fp4 = torch.randint(
        0,
        256,
        (experts_count, k, n // 2),
        device=device,
        dtype=torch.uint8,
    )
    w1_blockscale = swizzle_block_scale(
        torch.full(
            (experts_count, 2 * n, k // 16),
            0.015625,
            device=device,
            dtype=torch.float8_e4m3fn,
        )
    )
    w2_blockscale = swizzle_block_scale(
        torch.full(
            (experts_count, k, n // 16),
            0.015625,
            device=device,
            dtype=torch.float8_e4m3fn,
        )
    )
    scales = torch.ones(experts_count, device=device, dtype=torch.float32)
    experts = prepare_tp_moe_fp4_experts(
        a=x,
        a1_gscale=scales,
        w1_fp4=w1_fp4,
        w1_blockscale=w1_blockscale,
        w1_alphas=scales,
        a2_gscale=scales,
        w2_fp4=w2_fp4,
        w2_blockscale=w2_blockscale,
        w2_alphas=scales,
    )

    def run() -> torch.Tensor:
        clear_tp_moe_caches()
        output = run_tp_moe_fp4(
            a=x,
            experts=experts,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            input_scales_static=True,
        ).clone()
        torch.cuda.synchronize(device)
        return output

    monkeypatch.setenv("B12X_DYNAMIC_TILE_MN", "16x128")
    monkeypatch.delenv("B12X_DYNAMIC_DETERMINISTIC_OUTPUT", raising=False)
    atomic = run()
    monkeypatch.setenv("B12X_DYNAMIC_DETERMINISTIC_OUTPUT", "1")
    deterministic = run()
    repeat = run()

    assert torch.equal(deterministic, repeat)
    metrics = compare_to_reference(deterministic, atomic)
    assert metrics.max_abs <= 5e-4, metrics
    assert metrics.cos > 0.9999, metrics
