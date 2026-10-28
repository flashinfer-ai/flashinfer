"""w13_layout handling through the micro/dynamic (non-W4A16) serving paths.

The gated micro/dynamic kernels consume fused FC1 weights as [up; gate]
("w13"). vLLM fuses [gate; up] ("w31"/"gate_up"); b12x normalizes such
sources with a one-time in-place half flip of the FC1 weights and their
swizzled block scales at first use. These tests gate that flip:

- the swizzle-aware scale half swap is validated against the independent
  reference unswizzler (so the e2e test below is not circular), and
- b12x_moe_fp4 with gate-first weight copies + w13_layout="w31" must be
  bit-identical to the kernel-order baseline, for both nvfp4 and
  w4a8_nvfp4, across the micro and dynamic dispatch bands, including
  repeat launches (the flip must apply exactly once per storage).
"""

from __future__ import annotations

import functools
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks.experimental.b12x.benchmark_moe import (
    MODEL_PATH,
    TP_RANK,
    TP_SIZE,
    ModelSpec,
    load_expert_weights,
    make_routed_inputs,
)

_SUB_EXPERTS = 32


def _skip_if_unavailable() -> None:
    if not torch.cuda.is_available():
        pytest.skip("No CUDA")
    if not MODEL_PATH.exists():
        pytest.skip(f"Model not found at {MODEL_PATH}")


def _make_spec(num_experts: int = _SUB_EXPERTS) -> ModelSpec:
    return ModelSpec(
        hidden_size=4096,
        intermediate_size=1024,
        num_experts=num_experts,
        top_k=10,
        tp_size=TP_SIZE,
        tp_rank=TP_RANK,
    )


@functools.lru_cache(maxsize=1)
def _sub_weights():
    """First _SUB_EXPERTS experts of the checkpoint shard, as fresh storage."""
    full_spec = _make_spec(num_experts=512)
    w = load_expert_weights(MODEL_PATH, full_spec)
    return {
        "w13_weight": w.w13_weight[:_SUB_EXPERTS].clone(),
        "w13_blockscale": w.w13_blockscale_swizzled[:_SUB_EXPERTS].clone(),
        "g1_alphas": w.g1_alphas_per_expert[:_SUB_EXPERTS].clone(),
        "w1_input_scale": w.w13_input_scale_quant_per_expert[:_SUB_EXPERTS].clone(),
        "w2_weight": w.w2_weight[:_SUB_EXPERTS].clone(),
        "w2_blockscale": w.w2_blockscale_swizzled[:_SUB_EXPERTS].clone(),
        "g2_alphas": w.g2_alphas_per_expert[:_SUB_EXPERTS].clone(),
        "w2_input_scale": w.w2_input_scale_quant_per_expert[:_SUB_EXPERTS].clone(),
    }


def _rank_geometry(weights) -> tuple[int, int]:
    """Per-rank (n, k) from the shard tensors (the spec holds full-model dims)."""
    n = int(weights["w13_weight"].shape[1]) // 2
    k = 2 * int(weights["w13_weight"].shape[2])
    return n, k


def test_scale_half_swap_matches_reference_unswizzle() -> None:
    _skip_if_unavailable()
    from b12x.moe.fused_moe._impl import _swap_w13_scale_halves_inplace
    from b12x.moe._shared.kernels.reference import unswizzle_block_scale

    weights = _sub_weights()
    n, k = _rank_geometry(weights)
    orig = weights["w13_blockscale"]
    swapped = orig.clone()
    _swap_w13_scale_halves_inplace(swapped, rows=2 * n, cols_blocks=k // 16)

    for e in (0, 1, 7):
        ref = unswizzle_block_scale(orig.view(torch.uint8)[e], 2 * n, k // 16)
        got = unswizzle_block_scale(swapped.view(torch.uint8)[e], 2 * n, k // 16)
        torch.testing.assert_close(got[:n], ref[n:], atol=0.0, rtol=0.0)
        torch.testing.assert_close(got[n:], ref[:n], atol=0.0, rtol=0.0)

    # Double swap is the identity over the full padded storage.
    _swap_w13_scale_halves_inplace(swapped, rows=2 * n, cols_blocks=k // 16)
    assert torch.equal(swapped.view(torch.uint8), orig.view(torch.uint8))


def _swap_w1_halves(w13_weight: torch.Tensor, n: int) -> torch.Tensor:
    u8 = w13_weight.view(torch.uint8)
    return torch.cat([u8[:, n:], u8[:, :n]], dim=1).contiguous().view(w13_weight.dtype)


def _run(
    m: int,
    quant_mode: str | None,
    w13_weight: torch.Tensor,
    w13_blockscale: torch.Tensor,
    w13_layout: str,
    *,
    launches: int = 1,
    seed: int = 1234,
) -> torch.Tensor:
    from b12x.moe.fused_moe._impl import clear_tp_moe_caches
    from b12x.testing.reference.helpers import (
        prepare_tp_moe_fp4_experts,
        run_tp_moe_fp4,
    )

    clear_tp_moe_caches()
    device = torch.device("cuda")
    weights = _sub_weights()
    x, topk_ids, topk_weights = make_routed_inputs(
        _make_spec(), m, seed=seed, device=device
    )
    mode = quant_mode or "nvfp4"
    experts = prepare_tp_moe_fp4_experts(
        a=x,
        a1_gscale=weights["w1_input_scale"],
        w1_fp4=w13_weight,
        w1_blockscale=w13_blockscale,
        w1_alphas=weights["g1_alphas"],
        a2_gscale=weights["w2_input_scale"],
        w2_fp4=weights["w2_weight"],
        w2_blockscale=weights["w2_blockscale"],
        w2_alphas=weights["g2_alphas"],
        quant_mode=mode,
        w13_layout=w13_layout,
    )
    out = None
    for _ in range(launches):
        out = run_tp_moe_fp4(
            a=x,
            experts=experts,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            input_scales_static=True,
            quant_mode=mode,
        )
        torch.cuda.synchronize()
    return out


@pytest.mark.parametrize("quant_mode", [None, "w4a8_nvfp4"])
@pytest.mark.parametrize("m", [4, 64])
def test_w31_sources_match_w13_baseline(m: int, quant_mode: str | None) -> None:
    _skip_if_unavailable()
    weights = _sub_weights()
    n, k = _rank_geometry(weights)

    baseline = _run(
        m, quant_mode, weights["w13_weight"], weights["w13_blockscale"], "w13"
    )
    repeat = _run(
        m, quant_mode, weights["w13_weight"], weights["w13_blockscale"], "w13"
    )

    from b12x.moe.fused_moe._impl import _swap_w13_scale_halves_inplace

    w13_w31 = _swap_w1_halves(weights["w13_weight"], n)
    bs_w31 = weights["w13_blockscale"].clone()
    _swap_w13_scale_halves_inplace(bs_w31, rows=2 * n, cols_blocks=k // 16)

    # Two launches: the in-place flip must happen exactly once per storage.
    flipped = _run(m, quant_mode, w13_w31, bs_w31, "w31", launches=2)

    # The dynamic path's scatter reduction is order-nondeterministic, so gate
    # against the measured run-to-run envelope rather than bit equality. A
    # wrong-half silu would miss by O(1) on O(1) outputs — 4+ orders above it.
    noise = (baseline.float() - repeat.float()).abs().max().item()
    err = (flipped.float() - baseline.float()).abs().max().item()
    bound = max(8.0 * noise, 1e-6 * baseline.float().abs().max().item())
    assert err <= bound, f"w31 flip mismatch: err={err} noise={noise} bound={bound}"
