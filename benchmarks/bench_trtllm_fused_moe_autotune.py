#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark trtllm FP8 / FP4 MoE kernels.

Compares autotuned tactic vs default (first-valid) tactic latency for each
(quant_type, routing_mode, batch_size) combination.
"""

import torch
from flashinfer import autotune, RoutingMethodType
from flashinfer.autotuner import AutoTuner
from flashinfer.fused_moe import (
    Fp8QuantizationType,
    trtllm_fp8_block_scale_moe,
    trtllm_fp8_block_scale_routed_moe,
    trtllm_fp4_block_scale_moe,
    trtllm_fp4_block_scale_routed_moe,
)

# ---------------------------------------------------------------------------
# Config — edit to select what to benchmark
# ---------------------------------------------------------------------------

HIDDEN = 3072
INTERMEDIATE = 1536
NUM_EXPERTS = 256
LOCAL_NUM_EXPERTS = 256
TOPK = 8
EP_RANK = 0

TP_SIZE = 2
INTERMEDIATE_PER_PARTITION = INTERMEDIATE // TP_SIZE

BATCH_SIZES = [1, 8, 64, 128, 256, 512, 1024, 2048]

QUANT_TYPES = ["fp8", "fp4"]  # "fp8" and/or "fp4"
ROUTING_MODES = [True, False]  # True = pre-computed topk_ids, False = routing_logits

WARMUP = 100
ITERS = 1000

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_autotuner():
    AutoTuner.get().profiling_cache.clear()


def _collect_times(record_fn, iters):
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        record_fn()
        ends[i].record()
    torch.cuda.synchronize()
    return min(s.elapsed_time(e) for s, e in zip(starts, ends))


def min_ms_graph(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return _collect_times(g.replay, iters)


def _pack_topk(M, device):
    topk_ids = torch.randint(
        0, LOCAL_NUM_EXPERTS, (M, TOPK), dtype=torch.int32, device=device
    )
    raw_w = torch.rand(M, TOPK, device=device)
    weights = (raw_w / raw_w.sum(-1, keepdim=True)).to(torch.bfloat16)
    return (topk_ids << 16) | weights.view(torch.int16).to(torch.int32)


def _make_fp8_kernel(M, routed, device):
    N = INTERMEDIATE_PER_PARTITION
    hidden_states = (
        torch.randn(M, HIDDEN, device=device).clamp(-1, 1).to(torch.float8_e4m3fn)
    )
    hidden_states_scale = torch.ones(
        HIDDEN // 128, M, dtype=torch.float32, device=device
    )
    w1 = (
        torch.randn(LOCAL_NUM_EXPERTS, 2 * N, HIDDEN, device=device)
        .clamp(-1, 1)
        .to(torch.float8_e4m3fn)
    )
    w1_scale = torch.ones(
        LOCAL_NUM_EXPERTS,
        (2 * N) // 128,
        HIDDEN // 128,
        dtype=torch.float32,
        device=device,
    )
    w2 = (
        torch.randn(LOCAL_NUM_EXPERTS, HIDDEN, N, device=device)
        .clamp(-1, 1)
        .to(torch.float8_e4m3fn)
    )
    w2_scale = torch.ones(
        LOCAL_NUM_EXPERTS, HIDDEN // 128, N // 128, dtype=torch.float32, device=device
    )

    common = dict(
        routing_bias=None,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=w1,
        gemm1_weights_scale=w1_scale,
        gemm2_weights=w2,
        gemm2_weights_scale=w2_scale,
        num_experts=NUM_EXPERTS,
        top_k=TOPK,
        n_group=None,
        topk_group=None,
        intermediate_size=N,
        local_expert_offset=EP_RANK * LOCAL_NUM_EXPERTS,
        local_num_experts=LOCAL_NUM_EXPERTS,
        routed_scaling_factor=None,
        routing_method_type=RoutingMethodType.Renormalize.value,
        fp8_quantization_type=Fp8QuantizationType.DeepSeekFp8,
    )

    if routed:
        packed = _pack_topk(M, device)
        return lambda: trtllm_fp8_block_scale_routed_moe(topk_ids=packed, **common)
    else:
        routing_logits = torch.rand(M, NUM_EXPERTS, dtype=torch.bfloat16, device=device)
        return lambda: trtllm_fp8_block_scale_moe(
            routing_logits=routing_logits, **common
        )


def _make_fp4_kernel(M, routed, device):
    N = INTERMEDIATE_PER_PARTITION
    hidden_states = torch.randn(M, HIDDEN, dtype=torch.bfloat16, device=device)
    w1 = torch.empty(
        LOCAL_NUM_EXPERTS, 2 * N, HIDDEN // 2, dtype=torch.uint8, device=device
    )
    w1_scale = torch.empty(
        LOCAL_NUM_EXPERTS,
        2 * N,
        HIDDEN // 2 // 16,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    w2 = torch.empty(
        LOCAL_NUM_EXPERTS, HIDDEN, N // 2, dtype=torch.uint8, device=device
    )
    w2_scale = torch.empty(
        LOCAL_NUM_EXPERTS,
        HIDDEN,
        N // 2 // 16,
        dtype=torch.float8_e4m3fn,
        device=device,
    )

    common = dict(
        routing_bias=None,
        hidden_states=hidden_states,
        hidden_states_scale=None,
        gemm1_weights=w1,
        gemm1_weights_scale=w1_scale,
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=w2,
        gemm2_weights_scale=w2_scale,
        gemm2_bias=None,
        output1_scale_scalar=None,
        output1_scale_gate_scalar=None,
        output2_scale_scalar=None,
        num_experts=NUM_EXPERTS,
        top_k=TOPK,
        n_group=None,
        topk_group=None,
        intermediate_size=N,
        local_expert_offset=EP_RANK * LOCAL_NUM_EXPERTS,
        local_num_experts=LOCAL_NUM_EXPERTS,
        routed_scaling_factor=None,
        routing_method_type=RoutingMethodType.Renormalize.value,
    )

    if routed:
        packed = _pack_topk(M, device)
        return lambda: trtllm_fp4_block_scale_routed_moe(topk_ids=packed, **common)
    else:
        routing_logits = torch.rand(M, NUM_EXPERTS, dtype=torch.bfloat16, device=device)
        return lambda: trtllm_fp4_block_scale_moe(
            routing_logits=routing_logits, **common
        )


# ---------------------------------------------------------------------------
# Per-config benchmark
# ---------------------------------------------------------------------------


def run_benchmark(quant_type: str, routed: bool):
    device = "cuda"
    mode_str = "routed" if routed else "non_routed"
    make_kernel = _make_fp8_kernel if quant_type == "fp8" else _make_fp4_kernel

    print(f"\n{'=' * 66}")
    print(f"  {quant_type.upper()} MoE  [{mode_str}]")
    print(
        f"  hidden={HIDDEN}  intermediate={INTERMEDIATE}  "
        f"intermediate_per_partition={INTERMEDIATE_PER_PARTITION}  tp={TP_SIZE}"
    )
    print(f"  experts={NUM_EXPERTS}  topk={TOPK}  warmup={WARMUP}  iters={ITERS}")
    print(f"{'=' * 66}")

    # Build kernels for all batch sizes upfront
    kernels = {bs: make_kernel(bs, routed, device) for bs in BATCH_SIZES}

    # Measure before tuning (default tactic, no cache)
    lats_before = {}
    for bs in BATCH_SIZES:
        lats_before[bs] = min_ms_graph(kernels[bs])
    _reset_autotuner()

    # Tune once on batch_size=1; the autotuner sweeps all bucket sizes up to tune_max_num_tokens
    with autotune(tune_mode=True):
        kernels[1]()

    print(
        f"  {'batch_size':>12}  {'before (ms)':>13}  {'after (ms)':>12}  {'speedup':>9}"
    )
    print(f"  {'-' * 12}  {'-' * 13}  {'-' * 12}  {'-' * 9}")

    # Measure after tuning (cached best tactic for each bucket)
    for bs in BATCH_SIZES:
        lat_before = lats_before[bs]
        lat_after = min_ms_graph(kernels[bs])
        speedup = lat_before / lat_after
        print(f"  {bs:>12}  {lat_before:>13.3f}  {lat_after:>12.3f}  {speedup:>9.2f}x")
    _reset_autotuner()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for quant_type in QUANT_TYPES:
        for routed in ROUTING_MODES:
            run_benchmark(quant_type, routed)
