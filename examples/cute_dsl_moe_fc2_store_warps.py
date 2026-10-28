#!/usr/bin/env python3
# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Enable the store-warp epilogue of the CuTe-DSL NVFP4 MoE FC2 (GEMM2) kernel.

Run: python examples/cute_dsl_moe_fc2_store_warps.py [--tokens 15900]

FC2 with fused finalize writes one BF16 row per routed token with TMA bulk
reductions. For large token counts that write-back dominates the kernel. The
store-warp epilogue moves those writes off the critical path:

* ``num_store_warps`` dedicated warps issue the per-row bulk reductions; the
  epilogue warps hand each filled smem C tile to them through an mbarrier
  pipeline and immediately start the next tile.
* ``num_c_stage`` smem C tiles let the epilogue fill one while the store warps
  drain another (costs A/B pipeline stages).
* ``swizzle_size`` with ``raster_along_m=True`` visits a group of N tiles for
  every M tile, so the output columns being reduced at any time fit in L2.

It is opt-in through a 4th element on the Blackwell (SM100/SM103) GEMM2
sub-tactic; plain 3-element tactics keep the original epilogue:

    gemm1 = ((256, 256), (2, 1), False)
    gemm2 = ((256, 256), (2, 1), True, (2, 4, 4))  # (num_c_stage, num_store_warps, swizzle_size)
    out = cute_dsl_fused_moe(..., tactic=(256, gemm1, gemm2))

Measured with 384 experts, H=4096, I=768, top-k=20, 15900 tokens: FC2 went
from 0.79 to 0.72 ms on B300 and from ~0.89 to 0.79 ms on B200 (whole MoE
forward -4% / -5%). Non-fused output is bitwise identical to the default
epilogue; fused output differs only by BF16 atomic-add order (~2e-3 rel L2).
The autotuner searches the epilogues in BLACKWELL_GEMM2_EPILOGUE_CANDIDATES
(flashinfer/fused_moe/cute_dsl/tuner.py); pass a tactic to pin one. For a
sweep over token counts, see benchmarks/bench_cute_dsl_moe_fc2_epilogue.py.
"""

import argparse

import torch

from flashinfer import cute_dsl_fused_moe
from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
from flashinfer.fp4_quantization import fp4_quantize

SF_VEC_SIZE = 16
GEMM1 = ((256, 256), (2, 1), False)
BASELINE_TACTIC = (256, GEMM1, ((256, 256), (2, 1), False))
STORE_WARP_TACTIC = (256, GEMM1, ((256, 256), (2, 1), True, (2, 4, 4)))


def quantize_weight(w: torch.Tensor):
    """Pack [E, N, K] BF16 expert weights as NVFP4 with MMA-layout scales."""
    e, n, k = w.shape
    one = torch.ones(1, dtype=torch.float32, device=w.device)
    q, sf = fp4_quantize(
        w.reshape(e * n, k).contiguous(),
        global_scale=one,
        sf_vec_size=SF_VEC_SIZE,
        is_sf_swizzled_layout=True,
    )
    sf = convert_sf_to_mma_layout(sf, m=n, k=k, num_groups=e, sf_vec_size=SF_VEC_SIZE)
    return q.view(e, n, k // 2), sf


def make_inputs(tokens, experts, hidden, inter, top_k):
    dev = torch.device("cuda")
    torch.manual_seed(0)
    one = torch.ones(1, dtype=torch.float32, device=dev)
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=dev) / 10
    x_fp4, x_sf = fp4_quantize(
        x, global_scale=one, sf_vec_size=SF_VEC_SIZE, is_sf_swizzled_layout=False
    )
    # Random weights, so the FC1 [linear, gate] row interleave is irrelevant
    # here; real checkpoints need interleave_linear_and_gate(w1, 64, dim=1).
    w1 = torch.randn(experts, 2 * inter, hidden, dtype=torch.bfloat16, device=dev) / 10
    w2 = torch.randn(experts, hidden, inter, dtype=torch.bfloat16, device=dev) / 10
    w1_q, w1_sf = quantize_weight(w1)
    w2_q, w2_sf = quantize_weight(w2)
    logits = torch.randn(tokens, experts, device=dev)
    scales, selected = torch.topk(logits.softmax(dim=1), top_k, dim=1)
    return dict(
        x=x_fp4,
        x_sf=x_sf.unsqueeze(-1),
        token_selected_experts=selected.to(torch.int32),
        token_final_scales=(scales / scales.sum(dim=1, keepdim=True)).float(),
        w1_weight=w1_q,
        w1_weight_sf=w1_sf,
        w1_alpha=torch.ones(experts, dtype=torch.float32, device=dev),
        fc2_input_scale=one,
        w2_weight=w2_q,
        w2_weight_sf=w2_sf,
        w2_alpha=torch.ones(experts, dtype=torch.float32, device=dev),
        num_experts=experts,
        top_k=top_k,
    )


def bench_us(fn, iters=10):
    """(FC2 kernel, all kernels) GPU time per call in us, via the profiler."""
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    fc2 = total = 0.0
    for e in prof.key_averages():
        if e.device_type.name != "CUDA":
            continue
        total += e.device_time_total
        if "FinalizeFusion" in e.key:
            fc2 += e.device_time_total
    return fc2 / iters, total / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=15900)
    p.add_argument("--experts", type=int, default=384)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--inter", type=int, default=768)
    p.add_argument("--top-k", type=int, default=20)
    args = p.parse_args()

    inputs = make_inputs(args.tokens, args.experts, args.hidden, args.inter, args.top_k)

    def run(tactic):
        return cute_dsl_fused_moe(**inputs, quant_mode="w4a4", tactic=tactic)

    base = run(BASELINE_TACTIC).float()
    store = run(STORE_WARP_TACTIC).float()
    rel = ((store - base).norm() / base.norm()).item()
    print(f"{torch.cuda.get_device_name()}: store-warp vs baseline rel_l2={rel:.2e}")

    # Short alternating rounds: long back-to-back runs are skewed by power
    # throttling.
    for rnd in range(3):
        for name, tactic in (
            ("baseline", BASELINE_TACTIC),
            ("store-warp", STORE_WARP_TACTIC),
        ):
            fc2, total = bench_us(lambda: run(tactic))
            print(f"[{rnd}] {name:10s}: FC2 {fc2:6.0f} us, all kernels {total:6.0f} us")


if __name__ == "__main__":
    main()
