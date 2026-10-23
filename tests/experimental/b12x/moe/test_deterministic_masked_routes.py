"""Deterministic b12x MoE must not reuse routed rows of masked routes."""

from __future__ import annotations

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks.experimental.b12x.benchmark_ds4_moe import make_synthetic_mxfp4_moe
from b12x.testing.reference.helpers import (
    make_tp_moe_fp4_binding,
    prepare_tp_moe_fp4_experts,
    require_b12x,
)

_E = 16
_K = 5120
_N = 2048
_TOPK = 6


def _prepared_experts(a: torch.Tensor):
    weights = make_synthetic_mxfp4_moe(_E, _K, _N, seed=21, device=torch.device("cuda"))
    return prepare_tp_moe_fp4_experts(
        a=a,
        a1_gscale=weights["input_scale"],
        w1_fp4=weights["w13_fp4"],
        w1_blockscale=weights["w13_mx"],
        w1_alphas=weights["alphas"],
        a2_gscale=weights["input_scale"],
        w2_fp4=weights["w2_fp4"],
        w2_blockscale=weights["w2_mx"],
        w2_alphas=weights["alphas"],
        quant_mode="w4a8_mx",
    )


def _routed_inputs(m: int, seed: int):
    device = torch.device("cuda")
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    x = (torch.randn(m, _K, generator=gen, device=device) * 2.0).to(torch.bfloat16)
    logits = torch.randn(m, _E, generator=gen, device=device)
    topk_logits, topk_ids = torch.topk(logits, _TOPK, dim=-1)
    topk_weights = torch.softmax(topk_logits, dim=-1).float()
    return x, topk_ids.to(torch.int32), topk_weights


def test_masked_routes_do_not_reuse_previous_call_rows() -> None:
    """Routes with topk_ids == -1 have no producer, so their rows must be clear.

    The deterministic split writes one row per routed pair and skips routes
    without a local expert.  The fixed-order top-k reduction sums every routed
    row and does not consult topk_ids, so a reused plan must not feed an earlier
    call's values into rows whose routes are masked out.
    """
    require_b12x()
    from b12x.moe import fused_moe as fm

    m = 128
    x, topk_ids, topk_weights = _routed_inputs(m, seed=41)
    experts = _prepared_experts(x)
    output = torch.empty_like(x)

    with make_tp_moe_fp4_binding(
        a=x,
        experts=experts,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        output=output,
        deterministic_output=True,
        quant_mode="w4a8_mx",
    ) as binding:
        # First call runs every route and leaves its rows in the shared scratch.
        fm.run(binding=binding)
        torch.cuda.synchronize()

        gen = torch.Generator(device=topk_ids.device)
        gen.manual_seed(42)
        masked = torch.rand(topk_ids.shape, generator=gen, device=topk_ids.device) < 0.6
        topk_ids.masked_fill_(masked, -1)
        fm.run(binding=binding)
        torch.cuda.synchronize()
        masked_output = output.clone()

        all_masked = (topk_ids == -1).all(dim=1)
        assert bool(all_masked.any())
        rows = masked_output[all_masked]
        assert torch.equal(rows, torch.zeros_like(rows)), (
            "masked routes inherited routed rows from the previous call: "
            f"max_abs={rows.abs().max().item()}"
        )

        # A masked route must match a local expert that carries zero weight.
        neutral_weights = topk_weights.masked_fill(topk_ids < 0, 0)
        topk_ids.clamp_min_(0)
        topk_weights.copy_(neutral_weights)
        fm.run(binding=binding)
        torch.cuda.synchronize()

    assert torch.equal(masked_output, output), (
        "masked routes changed the result versus a zero-weight local expert: "
        f"max_abs={(masked_output - output).abs().max().item()}"
    )
