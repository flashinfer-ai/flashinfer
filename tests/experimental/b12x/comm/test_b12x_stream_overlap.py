from __future__ import annotations

import pytest
import torch

from benchmarks.experimental.b12x.benchmark_moe import MODEL_PATH, TP_RANK, TP_SIZE, ModelSpec, load_expert_weights
from b12x.moe.fused_moe._impl import (
    clear_tp_moe_caches,
)
from b12x.moe._shared.kernels.reference import compare_to_reference

from b12x.testing.reference.helpers import prepare_tp_moe_fp4_experts, require_b12x, run_tp_moe_fp4


def _require_model_weights() -> None:
    if not MODEL_PATH.exists():
        pytest.skip(f"Model not found at {MODEL_PATH}")


def _make_spec() -> ModelSpec:
    return ModelSpec(
        hidden_size=4096,
        intermediate_size=1024,
        num_experts=512,
        top_k=10,
        tp_size=TP_SIZE,
        tp_rank=TP_RANK,
    )


def _make_inputs(
    spec: ModelSpec,
    *,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    x = torch.randn(batch_size, spec.hidden_size, generator=gen, dtype=torch.float32)
    x = x.to(device=device, dtype=torch.bfloat16)
    topk_ids = torch.randint(
        low=0,
        high=spec.num_experts,
        size=(batch_size, spec.top_k),
        generator=gen,
        dtype=torch.int64,
    ).to(device=device)
    topk_weights = torch.rand(
        batch_size,
        spec.top_k,
        generator=gen,
        dtype=torch.float32,
    ).to(device=device)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    return x, topk_ids, topk_weights


def _run_once(
    *,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    weights,
) -> torch.Tensor:
    experts = prepare_tp_moe_fp4_experts(
        a=x,
        a1_gscale=weights.w13_input_scale_quant,
        w1_fp4=weights.w13_weight,
        w1_blockscale=weights.w13_blockscale_swizzled,
        w1_alphas=weights.g1_alphas,
        a2_gscale=weights.w2_input_scale_quant,
        w2_fp4=weights.w2_weight,
        w2_blockscale=weights.w2_blockscale_swizzled,
        w2_alphas=weights.g2_alphas,
    )
    return run_tp_moe_fp4(
        a=x,
        experts=experts,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    ).clone()


def _launch_with_alias_consumer(
    *,
    stream: torch.cuda.Stream,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    weights,
) -> torch.Tensor:
    with torch.cuda.stream(stream):
        experts = prepare_tp_moe_fp4_experts(
            a=x,
            a1_gscale=weights.w13_input_scale_quant,
            w1_fp4=weights.w13_weight,
            w1_blockscale=weights.w13_blockscale_swizzled,
            w1_alphas=weights.g1_alphas,
            a2_gscale=weights.w2_input_scale_quant,
            w2_fp4=weights.w2_weight,
            w2_blockscale=weights.w2_blockscale_swizzled,
            w2_alphas=weights.g2_alphas,
        )
        alias = run_tp_moe_fp4(
            a=x,
            experts=experts,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )
        sink = torch.empty_like(alias)
        sink.copy_(alias)
    return sink


def _assert_matches(actual: torch.Tensor, expected: torch.Tensor) -> None:
    metrics = compare_to_reference(actual, expected)
    assert metrics.max_abs <= 2e-3
    assert metrics.rmse <= 2e-5
    assert metrics.cos > 0.99989


def test_b12x_supports_overlapping_stream_launches() -> None:
    require_b12x()
    _require_model_weights()

    clear_tp_moe_caches()

    device = torch.device("cuda")
    spec = _make_spec()
    weights_a = load_expert_weights(MODEL_PATH, spec, layer_idx=0)
    weights_b = load_expert_weights(MODEL_PATH, spec, layer_idx=1)
    # Keep this on the compact-static path. The purpose of this test is explicit
    # lane safety across overlapping streams, not dynamic-path accuracy drift.
    xa, ida, wa = _make_inputs(spec, batch_size=1, seed=123, device=device)
    xb, idb, wb = _make_inputs(spec, batch_size=1, seed=456, device=device)

    torch.cuda.synchronize(device)
    ref_a = _run_once(
        x=xa,
        topk_ids=ida,
        topk_weights=wa,
        weights=weights_a,
    )
    ref_b = _run_once(
        x=xb,
        topk_ids=idb,
        topk_weights=wb,
        weights=weights_b,
    )
    torch.cuda.synchronize(device)

    stream_a = torch.cuda.Stream(device=device, priority=0)
    stream_b = torch.cuda.Stream(device=device, priority=0)
    with torch.cuda.stream(stream_a):
        out_a = _run_once(
            x=xa,
            topk_ids=ida,
            topk_weights=wa,
            weights=weights_a,
        )
    with torch.cuda.stream(stream_b):
        out_b = _run_once(
            x=xb,
            topk_ids=idb,
            topk_weights=wb,
            weights=weights_b,
        )
    torch.cuda.synchronize(device)

    stream_c = torch.cuda.Stream(device=device, priority=0)
    stream_d = torch.cuda.Stream(device=device, priority=0)
    sink_a = _launch_with_alias_consumer(
        stream=stream_c,
        x=xa,
        topk_ids=ida,
        topk_weights=wa,
        weights=weights_a,
    )
    sink_b = _launch_with_alias_consumer(
        stream=stream_d,
        x=xb,
        topk_ids=idb,
        topk_weights=wb,
        weights=weights_b,
    )
    torch.cuda.synchronize(device)

    _assert_matches(out_a, ref_a)
    _assert_matches(out_b, ref_b)
    _assert_matches(sink_a, ref_a)
    _assert_matches(sink_b, ref_b)
