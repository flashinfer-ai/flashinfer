"""
Sweep benchmark: B12xMoEWrapper vs cutlass_fused_moe (NVFP4) on SM120/121.

Charts the prefill/decode crossover so we can pin a default
``cutlass_prefill_threshold`` inside ``B12xMoEWrapper``.

For each ``num_tokens`` in the sweep we run both backends with their native
weight packings (built from independent BF16 baselines — perf is the only
metric, so we don't need bit-equal weights) and record median GPU time via
CUPTI.

Usage:
    python benchmarks/b12x_vs_cutlass_moe_sweep.py \\
        --hidden 6144 --intermediate 3072 --num_experts 128 --top_k 8 \\
        --activation relu2 \\
        --num_tokens 1 4 8 16 32 64 128 256 512 1024 2048 4096 8192 \\
        --out sweep.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from routines.moe import _create_nvfp4_moe_test_data, _activation_kwarg
from routines.moe_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    compute_routing,
)

import flashinfer  # noqa: F401
from flashinfer import (
    ActivationType,
    B12xMoEWrapper,
    cutlass_fused_moe,
    fp4_quantize,
)
from flashinfer.testing.utils import bench_gpu_time


def _round_up(x: int, y: int) -> int:
    return (x + y - 1) // y * y


def build_cutlass_nvfp4_weights(
    *, num_experts: int, hidden: int, intermediate: int, is_gated: bool, device
):
    """Build CUTLASS-NVFP4-format weights from a fresh bf16 baseline.

    Mirrors the inline path in benchmarks/routines/moe.py::testCutlassFusedMoe
    (variant="nvfp4"). Returns the tensors that ``cutlass_fused_moe`` expects.
    """
    e = num_experts
    n = intermediate
    k = hidden
    w1_n = (2 if is_gated else 1) * n
    quant_blocksize = 16

    w31_bf16 = (
        torch.randn(e, w1_n, k, dtype=torch.bfloat16, device=device) / 10
    ).contiguous()
    w2_bf16 = (
        torch.randn(e, k, n, dtype=torch.bfloat16, device=device) / 10
    ).contiguous()

    w1_q = torch.empty((e, w1_n, k // 2), device=device, dtype=torch.uint8)
    w2_q = torch.empty((e, k, n // 2), device=device, dtype=torch.uint8)
    w1_blockscale = torch.empty(
        (e, _round_up(w1_n, 128), _round_up(k // quant_blocksize, 4)),
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    w2_blockscale = torch.empty(
        (e, _round_up(k, 128), _round_up(n // quant_blocksize, 4)),
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    w1_gs = torch.empty((e,), device=device, dtype=torch.float32)
    w2_gs = torch.empty((e,), device=device, dtype=torch.float32)

    for ex in range(e):
        w1_amax = torch.abs(w31_bf16[ex]).max().to(torch.float32)
        w2_amax = torch.abs(w2_bf16[ex]).max().to(torch.float32)
        w1_gs[ex] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w1_amax
        w2_gs[ex] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w2_amax
        w1_q[ex], w1_blockscale[ex] = fp4_quantize(w31_bf16[ex], w1_gs[ex])
        w2_q[ex], w2_blockscale[ex] = fp4_quantize(w2_bf16[ex], w2_gs[ex])

    a1_gs = torch.ones((), device=device, dtype=torch.float32)
    a2_gs = torch.ones((), device=device, dtype=torch.float32)

    quant_scales = [
        a1_gs,
        w1_blockscale.view(torch.int32),
        1.0 / (a1_gs * w1_gs),
        a2_gs,
        w2_blockscale.view(torch.int32),
        1.0 / (a2_gs * w2_gs),
    ]
    return {
        "w1_q": w1_q,
        "w2_q": w2_q,
        "quant_scales": quant_scales,
        "a1_gs": a1_gs,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type=int, required=True)
    p.add_argument("--intermediate", type=int, required=True)
    p.add_argument("--num_experts", type=int, required=True)
    p.add_argument("--top_k", type=int, required=True)
    p.add_argument(
        "--activation",
        choices=["silu", "relu2"],
        default="relu2",
        help="silu = SwiGLU (gated), relu2 = ReLU2 (non-gated, e.g. Nemotron-Super)",
    )
    p.add_argument(
        "--num_tokens",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
    )
    p.add_argument("--dry_run_iters", type=int, default=5)
    p.add_argument("--repeat_iters", type=int, default=30)
    p.add_argument("--no_cuda_graph", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="Output CSV path")
    p.add_argument("--skip_cutlass", action="store_true")
    p.add_argument("--skip_b12x", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required")

    device = torch.device("cuda")
    use_cuda_graph = not args.no_cuda_graph
    is_gated = args.activation == "silu"
    activation_type = ActivationType.Swiglu if is_gated else ActivationType.Relu2

    cc_major, cc_minor = torch.cuda.get_device_capability(device)
    if (cc_major, cc_minor) not in ((12, 0), (12, 1)):
        print(
            f"[WARN] Detected SM{cc_major}{cc_minor}; b12x kernels target SM120/121.",
            file=sys.stderr,
        )

    print(
        f"[INFO] device={torch.cuda.get_device_name(device)} "
        f"(SM{cc_major}{cc_minor})  hidden={args.hidden}  intermediate={args.intermediate}  "
        f"num_experts={args.num_experts}  top_k={args.top_k}  activation={args.activation}"
    )

    max_m = max(args.num_tokens)

    # --- b12x weight packing (also gives us bf16 input shape; we'll re-roll x per m) ---
    torch.manual_seed(args.seed)
    b12x_data = _create_nvfp4_moe_test_data(
        num_tokens=max_m,
        hidden_size=args.hidden,
        intermediate_size=args.intermediate,
        num_experts=args.num_experts,
        num_local_experts=args.num_experts,
        top_k=args.top_k,
        device=device,
        backend="b12x",
        is_gated=is_gated,
    )

    # --- CUTLASS NVFP4 weight packing (independent bf16 baseline; perf-only) ---
    torch.manual_seed(args.seed + 1)
    cutlass_w = build_cutlass_nvfp4_weights(
        num_experts=args.num_experts,
        hidden=args.hidden,
        intermediate=args.intermediate,
        is_gated=is_gated,
        device=device,
    )

    # --- Single B12xMoEWrapper covers all m ≤ max_m via internal slicing ---
    moe = (
        None
        if args.skip_b12x
        else B12xMoEWrapper(
            num_experts=args.num_experts,
            top_k=args.top_k,
            hidden_size=args.hidden,
            intermediate_size=args.intermediate,
            use_cuda_graph=use_cuda_graph,
            max_num_tokens=max_m,
            activation=args.activation,
        )
    )

    rows = []
    print(
        f"{'m':>7} {'b12x_us':>10} {'cutlass_us':>11} {'cutlass_speedup':>16} {'winner':>8}"
    )
    for m in args.num_tokens:
        torch.manual_seed(args.seed + 100 + m)
        x_bf16 = (
            torch.randn(m, args.hidden, dtype=torch.bfloat16, device=device) / 10
        )
        routing_logits = torch.randn(m, args.num_experts, device=device)
        routing_weights, selected_experts = compute_routing(routing_logits, args.top_k)
        selected_experts = selected_experts.to(torch.int32)

        # ----- b12x -----
        b12x_us = float("nan")
        if moe is not None:
            b12x_args = (
                x_bf16,
                b12x_data["w1_weight"],
                b12x_data["w1_weight_sf"],
                b12x_data["w1_alpha"],
                b12x_data["fc2_input_scale"],
                b12x_data["w2_weight"],
                b12x_data["w2_weight_sf"],
                b12x_data["w2_alpha"],
                selected_experts,
                routing_weights,
            )

            def run_b12x(x, w1, w1sf, w1a, fc2s, w2, w2sf, w2a, te, tfs):
                return moe.run(
                    x=x,
                    w1_weight=w1,
                    w1_weight_sf=w1sf,
                    w1_alpha=w1a,
                    fc2_input_scale=fc2s,
                    w2_weight=w2,
                    w2_weight_sf=w2sf,
                    w2_alpha=w2a,
                    token_selected_experts=te,
                    token_final_scales=tfs,
                )

            b12x_times = bench_gpu_time(
                fn=run_b12x,
                dry_run_iters=args.dry_run_iters,
                repeat_iters=args.repeat_iters,
                sleep_after_run=False,
                enable_cupti=True,
                use_cuda_graph=use_cuda_graph,
                cold_l2_cache=True,
                input_args=b12x_args,
            )
            b12x_us = float(np.median(b12x_times))

        # ----- CUTLASS NVFP4 -----
        cutlass_us = float("nan")
        if not args.skip_cutlass:
            out = torch.empty(m, args.hidden, dtype=torch.bfloat16, device=device)
            x_q, x_sf = fp4_quantize(x_bf16, cutlass_w["a1_gs"])

            def run_cutlass(x, te, rw, w1q, w2q, o):
                return cutlass_fused_moe(
                    x,
                    te.to(torch.int),
                    rw,
                    w1q.contiguous().view(torch.long),
                    w2q.contiguous().view(torch.long),
                    torch.bfloat16,
                    quant_scales=cutlass_w["quant_scales"],
                    input_sf=x_sf,
                    output=o,
                    **_activation_kwarg(cutlass_fused_moe, activation_type),
                )

            cutlass_args = (
                x_q,
                selected_experts,
                routing_weights,
                cutlass_w["w1_q"],
                cutlass_w["w2_q"],
                out,
            )
            cutlass_times = bench_gpu_time(
                fn=run_cutlass,
                dry_run_iters=args.dry_run_iters,
                repeat_iters=args.repeat_iters,
                sleep_after_run=False,
                enable_cupti=True,
                use_cuda_graph=use_cuda_graph,
                cold_l2_cache=True,
                input_args=cutlass_args,
            )
            cutlass_us = float(np.median(cutlass_times))

        if np.isnan(b12x_us) or np.isnan(cutlass_us):
            speedup = float("nan")
            winner = "?"
        else:
            speedup = b12x_us / cutlass_us  # >1 ⇒ cutlass faster
            winner = "cutlass" if cutlass_us < b12x_us else "b12x"
        print(
            f"{m:>7d} {b12x_us:>10.2f} {cutlass_us:>11.2f} {speedup:>16.2f} {winner:>8}"
        )
        rows.append(
            {
                "num_tokens": m,
                "b12x_us": b12x_us,
                "cutlass_us": cutlass_us,
                "cutlass_speedup_over_b12x": speedup,
                "winner": winner,
                "hidden": args.hidden,
                "intermediate": args.intermediate,
                "num_experts": args.num_experts,
                "top_k": args.top_k,
                "activation": args.activation,
            }
        )

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[INFO] Wrote {args.out}")

    # Highlight crossover
    crossovers = []
    for i in range(1, len(rows)):
        prev, cur = rows[i - 1], rows[i]
        if prev["winner"] != cur["winner"] and "?" not in (prev["winner"], cur["winner"]):
            crossovers.append((prev["num_tokens"], cur["num_tokens"], cur["winner"]))
    if crossovers:
        print("[INFO] Crossover(s):")
        for lo, hi, w in crossovers:
            print(f"        {lo} → {hi}: switch to {w}")
    else:
        print("[INFO] No crossover observed in the swept range.")


if __name__ == "__main__":
    main()
