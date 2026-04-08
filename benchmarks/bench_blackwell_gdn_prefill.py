"""
Copyright (c) 2025 by FlashInfer team.

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

"""
Performance benchmark for Blackwell GDN (Gated Delta Network) prefill kernel.

Compares FlashInfer's SM100 GDN prefill against FLA baseline.

Usage:
  python bench_blackwell_gdn_prefill.py --sweep
  python bench_blackwell_gdn_prefill.py --batch-size 8 --seq-len 1024
  python bench_blackwell_gdn_prefill.py --varlen --cu-seqlens 0 512 1024 2048
"""

import argparse
import sys
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from flashinfer.gdn_prefill import chunk_gated_delta_rule
from flashinfer.testing import bench_gpu_time
from flashinfer.utils import is_sm100a_supported

try:
    from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd as fla_base

    _has_fla = True
except ImportError:
    _has_fla = False


def _make_inputs(
    total_len: int,
    num_seqs: int,
    num_qk_heads: int,
    num_v_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str = "cuda",
    use_initial_state: bool = True,
):
    """Create input tensors in FlashInfer 3D format (total_len, H, D)."""
    num_o_heads = max(num_qk_heads, num_v_heads)

    q = torch.randn(total_len, num_qk_heads, head_dim, dtype=dtype, device=device)
    k = F.normalize(
        torch.randn(
            total_len, num_qk_heads, head_dim, dtype=torch.float32, device=device
        ),
        p=2,
        dim=-1,
    ).to(dtype)
    v = torch.randn(total_len, num_v_heads, head_dim, dtype=dtype, device=device)
    g = F.logsigmoid(
        torch.rand(total_len, num_o_heads, dtype=torch.float32, device=device)
    )
    beta = torch.rand(
        total_len, num_o_heads, dtype=torch.float32, device=device
    ).sigmoid()

    h0 = None
    if use_initial_state:
        h0 = torch.randn(
            num_seqs,
            num_o_heads,
            head_dim,
            head_dim,
            dtype=torch.float32,
            device=device,
        )

    o = torch.empty(total_len, num_o_heads, head_dim, dtype=dtype, device=device)
    s_out = torch.empty(
        num_seqs,
        num_o_heads,
        head_dim,
        head_dim,
        dtype=torch.float32,
        device=device,
    )

    return q, k, v, g, beta, h0, o, s_out


def benchmark_fixlen(
    batch_size: int,
    seq_len: int,
    num_qk_heads: int,
    num_v_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.bfloat16,
    warmup_iters: int = 10,
    benchmark_iters: int = 100,
    use_initial_state: bool = True,
) -> dict:
    """Benchmark GDN with fixed-length sequences."""
    device = "cuda"
    total_len = batch_size * seq_len

    q, k, v, g, beta, h0, o, s_out = _make_inputs(
        total_len,
        batch_size,
        num_qk_heads,
        num_v_heads,
        head_dim,
        dtype,
        use_initial_state=use_initial_state,
    )
    cu_seqlens = torch.arange(
        0, total_len + 1, seq_len, dtype=torch.int64, device=device
    )

    def fn_gdn():
        chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            None,
            h0,
            True,
            cu_seqlens,
            False,
            o,
            s_out,
        )

    gdn_times = bench_gpu_time(
        fn_gdn,
        enable_cupti=True,
        dry_run_iters=warmup_iters,
        repeat_iters=benchmark_iters,
    )
    gdn_ms = float(np.median(gdn_times))

    result = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "num_qk_heads": num_qk_heads,
        "num_v_heads": num_v_heads,
        "head_dim": head_dim,
        "gdn_ms": gdn_ms,
    }

    # FLA baseline (only when qk_heads == v_heads, FLA doesn't support GVA)
    if _has_fla and num_qk_heads == num_v_heads:
        # FLA expects 4D (B, T, H, D)
        q4 = q.view(batch_size, seq_len, num_qk_heads, head_dim)
        k4 = k.view(batch_size, seq_len, num_qk_heads, head_dim)
        v4 = v.view(batch_size, seq_len, num_v_heads, head_dim)
        g4 = g.view(batch_size, seq_len, num_v_heads)
        beta4 = beta.view(batch_size, seq_len, num_v_heads)

        def fn_fla():
            fla_base(
                q4, k4, v4, g4, beta4, None, initial_state=h0, output_final_state=True
            )

        fla_times = bench_gpu_time(
            fn_fla,
            enable_cupti=True,
            dry_run_iters=warmup_iters,
            repeat_iters=benchmark_iters,
        )
        fla_ms = float(np.median(fla_times))
        result["fla_ms"] = fla_ms
        result["speedup"] = fla_ms / gdn_ms if gdn_ms > 0 else float("nan")

    return result


def benchmark_varlen(
    cu_seqlens: List[int],
    num_qk_heads: int,
    num_v_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.bfloat16,
    warmup_iters: int = 10,
    benchmark_iters: int = 100,
) -> dict:
    """Benchmark GDN with variable-length sequences."""
    device = "cuda"
    total_len = cu_seqlens[-1]
    num_seqs = len(cu_seqlens) - 1

    q, k, v, g, beta, h0, o, s_out = _make_inputs(
        total_len,
        num_seqs,
        num_qk_heads,
        num_v_heads,
        head_dim,
        dtype,
    )
    cu_seqlens_t = torch.tensor(cu_seqlens, dtype=torch.int64, device=device)

    def fn_gdn():
        chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            None,
            h0,
            True,
            cu_seqlens_t,
            False,
            o,
            s_out,
        )

    gdn_times = bench_gpu_time(
        fn_gdn,
        enable_cupti=True,
        dry_run_iters=warmup_iters,
        repeat_iters=benchmark_iters,
    )
    gdn_ms = float(np.median(gdn_times))

    return {
        "num_seqs": num_seqs,
        "total_len": total_len,
        "avg_seq_len": total_len // num_seqs,
        "num_qk_heads": num_qk_heads,
        "num_v_heads": num_v_heads,
        "head_dim": head_dim,
        "gdn_ms": gdn_ms,
    }


def print_results_table(results: List[dict], title: str = "Benchmark Results"):
    """Print benchmark results in a formatted table."""
    if not results:
        return

    print(f"\n{'=' * 90}")
    print(f"  {title}")
    print(f"{'=' * 90}")

    # Collect all keys across all results for consistent columns
    keys = []
    seen = set()
    for r in results:
        for key in r:
            if key not in seen:
                keys.append(key)
                seen.add(key)

    widths = {}
    for key in keys:
        max_len = len(key)
        for r in results:
            val = r.get(key, "")
            if isinstance(val, float):
                val_str = f"{val:.3f}" if val < 100 else f"{val:.1f}"
            else:
                val_str = str(val)
            max_len = max(max_len, len(val_str))
        widths[key] = max_len + 2

    header = " | ".join(f"{key:^{widths[key]}}" for key in keys)
    print(header)
    print("-" * len(header))

    for r in results:
        row = []
        for key in keys:
            val = r.get(key, "")
            if isinstance(val, float):
                val_str = f"{val:.3f}" if val < 100 else f"{val:.1f}"
            else:
                val_str = str(val)
            row.append(f"{val_str:^{widths[key]}}")
        print(" | ".join(row))

    print(f"{'=' * 90}\n")


def run_sweep(warmup_iters: int = 10, benchmark_iters: int = 100):
    """Run the standard sweep matching PR #2742 configurations."""
    head_dim = 128

    fixlen_configs = [
        # (batch_size, seq_len, num_qk_heads, num_v_heads)
        (1, 512, 96, 96),
        (1, 1024, 96, 96),
        (1, 4096, 96, 96),
        (1, 8192, 96, 96),
        (9, 512, 32, 32),
        (9, 1024, 32, 32),
        (9, 4096, 32, 32),
        (9, 8192, 32, 32),
        (33, 512, 32, 32),
        (33, 1024, 32, 32),
        (33, 4096, 32, 32),
        (33, 8192, 32, 32),
        (1, 512, 148, 148),
        (1, 1024, 148, 148),
        (1, 4096, 148, 148),
        (1, 8192, 148, 148),
    ]

    print(f"\n{'#' * 80}")
    print("  BLACKWELL GDN PREFILL BENCHMARK SWEEP")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    if _has_fla:
        print("  FLA baseline: available")
    else:
        print("  FLA baseline: not installed (pip install flash-linear-attention)")
    print(f"{'#' * 80}")

    # Fixed-length benchmarks
    fixlen_results = []
    for i, (bs, sl, nqk, nv) in enumerate(fixlen_configs):
        label = f"[{i + 1}/{len(fixlen_configs)}] bs={bs}, sl={sl}, nqk={nqk}, nv={nv}"
        print(f"  Running fixlen {label} ...", end="", flush=True)
        try:
            result = benchmark_fixlen(
                bs,
                sl,
                nqk,
                nv,
                head_dim,
                warmup_iters=warmup_iters,
                benchmark_iters=benchmark_iters,
            )
            fixlen_results.append(result)
            msg = f" GDN: {result['gdn_ms']:.3f} ms"
            if "fla_ms" in result:
                msg += f"  FLA: {result['fla_ms']:.3f} ms  ({result['speedup']:.2f}x)"
            print(msg)
        except Exception as e:
            print(f" FAILED: {e}")
        torch.cuda.empty_cache()

    print_results_table(fixlen_results, "Fixed-Length Results")

    # Variable-length benchmarks (same configs, uniform seqlens)
    varlen_results = []
    for i, (bs, sl, nqk, nv) in enumerate(fixlen_configs):
        cu_seqlens = [sl * j for j in range(bs + 1)]
        num_seqs = bs
        total_len = cu_seqlens[-1]
        label = f"[{i + 1}/{len(fixlen_configs)}] seqs={num_seqs}, total={total_len}, nqk={nqk}, nv={nv}"
        print(f"  Running varlen {label} ...", end="", flush=True)
        try:
            result = benchmark_varlen(
                cu_seqlens,
                nqk,
                nv,
                head_dim,
                warmup_iters=warmup_iters,
                benchmark_iters=benchmark_iters,
            )
            varlen_results.append(result)
            print(f" GDN: {result['gdn_ms']:.3f} ms")
        except Exception as e:
            print(f" FAILED: {e}")
        torch.cuda.empty_cache()

    print_results_table(varlen_results, "Variable-Length Results")


def main():
    parser = argparse.ArgumentParser(
        description="Blackwell GDN Prefill Benchmark (SM100+)"
    )
    parser.add_argument("--batch-size", "-b", type=int, default=4)
    parser.add_argument("--seq-len", "-t", type=int, default=4096)
    parser.add_argument("--num-qk-heads", "-nqk", type=int, default=32)
    parser.add_argument("--num-v-heads", "-nv", type=int, default=32)
    parser.add_argument("--head-dim", "-d", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--varlen", action="store_true", help="Variable-length mode")
    parser.add_argument(
        "--cu-seqlens",
        type=int,
        nargs="+",
        default=None,
        help="Cumulative sequence lengths for varlen (e.g. 0 512 1024 2048)",
    )
    parser.add_argument("--sweep", action="store_true", help="Run full sweep")
    args = parser.parse_args()

    device = torch.device("cuda")
    if not is_sm100a_supported(device):
        print("Error: This benchmark requires SM100+ (Blackwell) GPU.")
        sys.exit(1)

    if args.head_dim != 128:
        print(f"Error: head_dim must be 128, got {args.head_dim}")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print("  Blackwell GDN Prefill Benchmark")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'=' * 60}")

    if args.sweep:
        run_sweep(warmup_iters=args.warmup, benchmark_iters=args.iters)
        return

    if args.varlen:
        if args.cu_seqlens is not None:
            cu_seqlens = args.cu_seqlens
        else:
            cu_seqlens = [args.seq_len * i for i in range(args.batch_size + 1)]
        result = benchmark_varlen(
            cu_seqlens,
            args.num_qk_heads,
            args.num_v_heads,
            args.head_dim,
            warmup_iters=args.warmup,
            benchmark_iters=args.iters,
        )
        print_results_table([result], "Variable-Length Benchmark")
    else:
        result = benchmark_fixlen(
            args.batch_size,
            args.seq_len,
            args.num_qk_heads,
            args.num_v_heads,
            args.head_dim,
            warmup_iters=args.warmup,
            benchmark_iters=args.iters,
        )
        print_results_table([result], "Fixed-Length Benchmark")


if __name__ == "__main__":
    main()
