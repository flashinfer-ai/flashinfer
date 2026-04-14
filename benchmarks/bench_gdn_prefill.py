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
Performance benchmark for GDN (Gated Delta Network) prefill kernel.

Compares FlashInfer GDN prefill against FLA baseline across
Qwen3.5 family model configurations.

Usage:
  python bench_gdn_prefill.py
  python bench_gdn_prefill.py --warmup 10 --iters 100
"""

import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F

from flashinfer.gdn_prefill import chunk_gated_delta_rule
from flashinfer.testing import bench_gpu_time
from flashinfer.utils import get_compute_capability

try:
    from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd as fla_gdn

    _has_fla = True
except ImportError:
    _has_fla = False

HEAD_CONFIGS = [
    # (h_qk, h_v, d, label)
    # Qwen3.5-397B and 122B (h_k=16, h_v=64, d=128) under different TP
    (2, 8, 128, "397B/122B TP8"),
    (4, 16, 128, "397B/122B TP4"),
    (8, 32, 128, "397B/122B TP2"),
    (16, 64, 128, "397B/122B TP1"),
    # Qwen3.5-35B, 9B and 4B (h_k=16, h_v=32, d=128)
    (16, 32, 128, "35B/9B/4B TP1"),
    # Qwen3.5-27B (h_k=16, h_v=48, d=128)
    (16, 48, 128, "27B TP1"),
    # Qwen3.5-2B and 0.8B (h_k=16, h_v=16, d=128)
    (16, 16, 128, "2B/0.8B TP1"),
    # Symmetric heads
    (32, 32, 128, "Sym h32"),
]

SEQ_CONFIGS = [
    # (cu_seqlen_endpoints, label)
    # endpoints are cumulative positions (leading 0 added automatically)
    ((8192,), "1x8192"),
    ((4096,), "1x4096"),
    ((2048,), "1x2048"),
    ((1024 * 6, 8192), "6144+2048"),
    ((1024 * 4, 8192), "4096+4096"),
    ((1024 * 2, 8192), "2048+6144"),
    ((1024 * 1, 8192), "1024+7168"),
    ((2048, 2048 * 2, 2048 * 3, 8192), "2048x4"),
    (tuple(1024 * (i + 1) for i in range(8)), "1024x8"),
]


def _gdn_tflops(total_tokens, h_v, d, time_ms):
    """Calculate TFLOPS: 2 GEMMs (kv outer product + q@state) per token per head."""
    flops = 2 * 2 * total_tokens * h_v * d * d
    return flops / time_ms / 1e9


def bench_fi(endpoints, h_qk, h_v, d, warmup, iters):
    """Benchmark FlashInfer GDN prefill."""
    device = "cuda"
    dtype = torch.float16
    N = len(endpoints)
    T = endpoints[-1]
    cu_seqlens = torch.tensor([0] + list(endpoints), dtype=torch.int64, device=device)

    q = torch.randn((T, h_qk, d), dtype=dtype, device=device)
    k = F.normalize(
        torch.randn(T, h_qk, d, dtype=torch.float32, device=device), p=2, dim=-1
    ).to(dtype)
    v = torch.randn((T, h_v, d), dtype=dtype, device=device)
    g = F.logsigmoid(torch.rand(T, h_v, dtype=torch.float32, device=device))
    beta = torch.rand(T, h_v, dtype=torch.float32, device=device).sigmoid()
    h0 = torch.randn((N, h_v, d, d), dtype=torch.float32, device=device)
    state_out = torch.zeros_like(h0)

    fn = lambda: chunk_gated_delta_rule(
        q, k, v, g, beta, None, h0, True, cu_seqlens, False, None, state_out
    )
    times = bench_gpu_time(
        fn, enable_cupti=True, dry_run_iters=warmup, repeat_iters=iters
    )
    torch.cuda.empty_cache()
    return np.average(times)


def bench_fla(endpoints, h_qk, h_v, d, warmup, iters):
    """Benchmark FLA baseline."""
    device = "cuda"
    dtype = torch.float16
    h = h_v
    N = len(endpoints)
    T = endpoints[-1]
    cu_seqlens = torch.tensor([0] + list(endpoints), dtype=torch.int32, device=device)

    q = torch.randn((1, T, h, d), dtype=dtype, device=device)
    k = F.normalize(
        torch.randn(1, T, h, d, dtype=torch.float32, device=device), p=2, dim=-1
    ).to(dtype)
    v = torch.randn((1, T, h_v, d), dtype=dtype, device=device)
    g = F.logsigmoid(torch.rand(1, T, h_v, dtype=torch.float32, device=device))
    beta = torch.rand(1, T, h_v, dtype=torch.float32, device=device).sigmoid()
    h0 = torch.randn((N, h_v, d, d), dtype=torch.float32, device=device)

    fn = lambda: fla_gdn(
        q,
        k,
        v,
        g,
        beta,
        None,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    times = bench_gpu_time(
        fn, enable_cupti=True, dry_run_iters=warmup, repeat_iters=iters
    )
    torch.cuda.empty_cache()
    return np.average(times)


def main():
    parser = argparse.ArgumentParser(description="Benchmark GDN Prefill Kernel")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda")
    major, minor = get_compute_capability(device)
    if major < 9:
        print(f"GDN requires SM90+, got SM{major}{minor}")
        sys.exit(1)

    arch_label = {9: "Hopper (SM90)", 10: "Blackwell (SM100)"}.get(
        major, f"SM{major}{minor}"
    )

    if not _has_fla:
        print("Error: FLA not installed. Run: pip install flash-linear-attention")
        sys.exit(1)

    print(f"\nGPU: {torch.cuda.get_device_name(0)} [{arch_label}]")
    print("Models: Qwen3.5 family (397B, 122B, 35B, 27B, 9B, 4B, 2B, 0.8B), d=128")
    print()
    fi_col = f"FI {arch_label}"
    header = (
        f"{'Heads':<15s}  {'Seqlens':<16s}  {'h_qk':>4s} {'h_v':>4s}"
        f"  {fi_col:>22s}  {'TFLOPS':>7s}"
        f"  {'FLA/Triton':>10s}  {'Speedup':>8s}"
    )
    print(header)
    print("-" * len(header))

    for h_qk, h_v, d, h_label in HEAD_CONFIGS:
        for endpoints, s_label in SEQ_CONFIGS:
            T = endpoints[-1]
            fi_ms = bench_fi(endpoints, h_qk, h_v, d, args.warmup, args.iters)
            fla_ms = bench_fla(endpoints, h_qk, h_v, d, args.warmup, args.iters)
            tflops = _gdn_tflops(T, h_v, d, fi_ms)
            speedup = fla_ms / fi_ms
            marker = "+" if speedup > 1.0 else "-"
            print(
                f"{h_label:<15s}  {s_label:<16s}  {h_qk:>4d} {h_v:>4d}"
                f"  {fi_ms:>21.3f}ms  {tflops:>6.1f}"
                f"  {fla_ms:>9.3f}ms  {speedup:>7.2f}x {marker}"
            )
        print()


if __name__ == "__main__":
    main()
