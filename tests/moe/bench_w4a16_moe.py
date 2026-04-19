"""Benchmark W4A16 (wMXFP4 x BF16) MoE kernel on H20 (SM90).

Configs:
  - Small bs (4): TP mode — full experts, split intermediate
  - Large bs (1024): EP mode — fewer local experts, full intermediate
"""

import torch
import torch.nn.functional as F

from flashinfer.utils import get_compute_capability


def bench_w4a16_moe(
    batch_size,
    hidden_size,
    num_experts,
    top_k,
    intermediate_size,
    num_warmup=10,
    num_iters=50,
    label="",
):
    import flashinfer.fused_moe as fused_moe

    torch.manual_seed(42)
    device = torch.device("cuda")

    e = num_experts
    m = batch_size
    n = intermediate_size
    k = hidden_size

    # BF16 activation
    x = torch.randn(m, k, dtype=torch.bfloat16, device=device)

    # MXFP4 weights (random uint8 packed)
    w1 = torch.randint(0, 256, (e, 2 * n, k // 2), device=device, dtype=torch.uint8)
    w2 = torch.randint(0, 256, (e, k, n // 2), device=device, dtype=torch.uint8)

    # MXFP4 scales
    w1_scale = torch.randint(
        118, 123, (e, 2 * n, k // 32), device=device, dtype=torch.uint8
    )
    w2_scale = torch.randint(
        118, 123, (e, k, n // 32), device=device, dtype=torch.uint8
    )

    # Routing
    router_logits = torch.randn(m, e, dtype=torch.bfloat16, device=device)
    routing_weights, selected_experts = torch.topk(
        F.softmax(router_logits.float(), dim=-1), top_k, dim=-1
    )
    routing_weights = (routing_weights / routing_weights.sum(dim=-1, keepdim=True)).to(
        torch.float32
    )

    # Weight + scale interleave runs once at weight-load time, not per-iter.
    # Mimics how a serving runtime would preprocess weights before the kernel
    # sees them (see TensorRT-LLM PR #12451). We do NOT count this against the
    # per-iteration GEMM time.
    w1 = fused_moe.interleave_moe_weights_for_hopper_mixed_gemm(w1, "fp4")
    w2 = fused_moe.interleave_moe_weights_for_hopper_mixed_gemm(w2, "fp4")
    w1_scale = fused_moe.interleave_moe_scales_for_hopper_mixed_gemm(w1_scale)
    w2_scale = fused_moe.interleave_moe_scales_for_hopper_mixed_gemm(w2_scale)
    torch.cuda.synchronize()

    quant_scales = [w1_scale.view(torch.int32), w2_scale.view(torch.int32)]
    output = torch.zeros_like(x)

    def run():
        fused_moe.cutlass_fused_moe(
            x,
            selected_experts.to(torch.int),
            routing_weights,
            w1,
            w2,
            torch.bfloat16,
            quant_scales=quant_scales,
            use_w4_group_scaling=True,
            output=output,
        )

    # Warmup
    for _ in range(num_warmup):
        run()
    torch.cuda.synchronize()

    # Benchmark
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    for i in range(num_iters):
        start_events[i].record()
        run()
        end_events[i].record()
    torch.cuda.synchronize()

    times_ms = [
        s.elapsed_time(e) for s, e in zip(start_events, end_events, strict=True)
    ]
    times_ms.sort()
    # Use median
    median_ms = times_ms[len(times_ms) // 2]
    p10_ms = times_ms[int(len(times_ms) * 0.1)]
    p90_ms = times_ms[int(len(times_ms) * 0.9)]

    # FLOPS: 2 GEMMs per token per expert
    # GEMM1: M*2N*K, GEMM2: M*K*N (per active expert)
    active_tokens = m * top_k
    flops_gemm1 = 2 * active_tokens * (2 * n) * k
    flops_gemm2 = 2 * active_tokens * k * n
    total_flops = flops_gemm1 + flops_gemm2
    tflops = total_flops / (median_ms * 1e-3) / 1e12

    print(
        f"  {label:40s} | bs={m:5d} e={e:4d} k={top_k} h={k:5d} n={n:5d} | "
        f"median={median_ms:8.3f}ms  p10={p10_ms:8.3f}ms  p90={p90_ms:8.3f}ms | "
        f"{tflops:7.2f} TFLOPS"
    )
    return median_ms, tflops


def main():
    cc = get_compute_capability(torch.device("cuda"))
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}, SM{cc[0]}{cc[1]}")
    print(f"{'=' * 120}")

    # =========================================================================
    # Config: Qwen3-MoE-like (hidden=4096, inter=2048, experts=256, topk=6)
    # =========================================================================
    H, N_full, E_full, K = 4096, 2048, 256, 6

    print("\n--- Small BS (decode-like, TP mode) ---")
    print("    TP splits intermediate_size, keeps all experts")
    for tp in [1, 2, 4, 8]:
        n_tp = N_full // tp
        if n_tp < 64:
            continue
        bench_w4a16_moe(
            batch_size=4,
            hidden_size=H,
            num_experts=E_full,
            top_k=K,
            intermediate_size=n_tp,
            label=f"TP{tp} (inter={n_tp})",
        )

    print("\n--- Large BS (prefill-like, EP mode) ---")
    print("    EP splits experts, keeps full intermediate_size")
    for ep in [1, 2, 4, 8]:
        e_ep = E_full // ep
        topk = min(K, e_ep)
        bench_w4a16_moe(
            batch_size=1024,
            hidden_size=H,
            num_experts=e_ep,
            top_k=topk,
            intermediate_size=N_full,
            label=f"EP{ep} (experts={e_ep}, topk={topk})",
        )

    # =========================================================================
    # Batch size sweep
    # =========================================================================
    print(f"\n--- BS sweep (full config: e={E_full}, h={H}, n={N_full}, topk={K}) ---")
    for bs in [1, 4, 16, 64, 256, 1024, 4096, 16384]:
        bench_w4a16_moe(
            batch_size=bs,
            hidden_size=H,
            num_experts=E_full,
            top_k=K,
            intermediate_size=N_full,
            label=f"bs={bs}",
        )


if __name__ == "__main__":
    main()
