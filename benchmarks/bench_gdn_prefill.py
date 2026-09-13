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

The default timing is CUDA events around each call, so host dispatch is inside
the number. Pass --use-cuda-graph to replay a captured graph instead and compare
kernels alone.

The cuDNN column pays one conversion the native path does not: FlashInfer's
public alpha is linear space and cuDNN's GDN engine takes a log gate, so the
wrapper runs a .log() inside the timed call. Every column passes
output_state=None so no backend is charged a state copy another avoids.

Usage:
  python bench_gdn_prefill.py
  python bench_gdn_prefill.py --warmup 10 --iters 100
"""

import argparse
import sys
import time
import gc

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
    fla_gdn = None
    _has_fla = False


def _cudnn_rejection_reason(device):
    """Why the cuDNN backend cannot run this benchmark, or None if it can."""
    tokens, heads, dim = 64, 1, 128
    qkv = dict(dtype=torch.float16, device=device)
    try:
        chunk_gated_delta_rule(
            torch.randn(tokens, heads, dim, **qkv),
            torch.randn(tokens, heads, dim, **qkv),
            torch.randn(tokens, heads, dim, **qkv),
            cu_seqlens=torch.tensor([0, tokens], dtype=torch.int32, device=device),
            backend="cudnn",
        )
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


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
    ((65536,), "1x65536"),
    ((32768,), "1x32768"),
    ((16384,), "1x16384"),
    ((8192,), "1x8192"),
    ((4096,), "1x4096"),
    ((2048,), "1x2048"),
    ((1024 * 6, 8192), "6144+2048"),
    ((1024 * 4, 8192), "4096+4096"),
    ((1024 * 2, 8192), "2048+6144"),
    ((1024 * 1, 8192), "1024+7168"),
    ((2048, 2048 * 2, 2048 * 3, 8192), "2048x4"),
    (tuple(1024 * (i + 1) for i in range(8)), "1024x8"),
    (tuple(8192 * (i + 1) for i in range(8)), "8192x8"),
    (tuple(8192 * (i + 1) for i in range(16)), "8192x16"),
    (tuple(8192 * (i + 1) for i in range(32)), "8192x32"),
]

DRIFT_THRESHOLD = 0.02


def _gdn_tflops(total_tokens, h_v, d, time_ms):
    """Calculate TFLOPS: 2 GEMMs (kv outer product + q@state) per token per head."""
    flops = 2 * 2 * total_tokens * h_v * d * d
    return flops / time_ms / 1e9


def get_num_rotating_buffers(num_iters: int, q, k, v) -> int:
    """Heuristic for number of rotating buffers to use based on total memory footprint."""
    nbytes = q.nbytes + k.nbytes + v.nbytes
    total_bytes = 4 * 1024 * 1024 * 1024  # 4 GB
    # Assume 4 GB per buffer is a reasonable heuristic; adjust as needed.
    return max(1, min(num_iters, (total_bytes + nbytes - 1) // nbytes))


def bench_fi(args, endpoints, h_qk, h_v, d, backend="auto"):
    """Benchmark FlashInfer GDN prefill on the given public API backend."""
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
    # FlashInfer's g is the linear-space forget gate alpha in (0, 1)
    # ("defaults to all ones" = no decay). Log-space gates (e.g. logsigmoid)
    # are out of domain and produce NaN outputs/state.
    g = torch.rand(T, h_v, dtype=torch.float32, device=device).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    beta = torch.rand(T, h_v, dtype=torch.float32, device=device).sigmoid()
    h0 = torch.randn((N, h_v, d, d), dtype=torch.float32, device=device)

    num_buffer = get_num_rotating_buffers(args.iters, q, k, v)
    q = [q.clone() for _ in range(num_buffer)]
    k = [k.clone() for _ in range(num_buffer)]
    v = [v.clone() for _ in range(num_buffer)]
    rotation_buffer_idx = 0

    def fn():
        nonlocal rotation_buffer_idx
        chunk_gated_delta_rule(
            q[rotation_buffer_idx % num_buffer],
            k[rotation_buffer_idx % num_buffer],
            v[rotation_buffer_idx % num_buffer],
            g,
            beta,
            None,
            h0,
            True,
            cu_seqlens,
            False,
            backend=backend,
        )
        rotation_buffer_idx += 1

    times = bench_gpu_time(
        fn,
        enable_cupti=args.use_cupti,
        dry_run_iters=args.warmup,
        repeat_iters=args.iters,
        use_cuda_graph=args.use_cuda_graph,
    )
    torch.cuda.empty_cache()
    return float(np.median(times))


def bench_fla(args, endpoints, h_qk, h_v, d):
    """Benchmark FLA baseline."""
    device = "cuda"
    dtype = torch.float16
    N = len(endpoints)
    T = endpoints[-1]
    cu_seqlens = torch.tensor([0] + list(endpoints), dtype=torch.int32, device=device)

    q = torch.randn((1, T, h_qk, d), dtype=dtype, device=device)
    k = F.normalize(
        torch.randn(1, T, h_qk, d, dtype=torch.float32, device=device), p=2, dim=-1
    ).to(dtype)
    v = torch.randn((1, T, h_v, d), dtype=dtype, device=device)
    g = (
        torch.rand(1, T, h_v, dtype=torch.float32, device=device)
        .clamp_min(torch.finfo(torch.float32).tiny)
        .log()
    )
    beta = torch.rand(1, T, h_v, dtype=torch.float32, device=device).sigmoid()
    h0 = torch.randn((N, h_v, d, d), dtype=torch.float32, device=device)

    num_buffer = get_num_rotating_buffers(args.iters, q, k, v)
    q = [q.clone() for _ in range(num_buffer)]
    k = [k.clone() for _ in range(num_buffer)]
    v = [v.clone() for _ in range(num_buffer)]

    rotation_buffer_idx = 0

    def fn():
        nonlocal rotation_buffer_idx
        fla_gdn(
            q[rotation_buffer_idx % num_buffer],
            k[rotation_buffer_idx % num_buffer],
            v[rotation_buffer_idx % num_buffer],
            g,
            beta,
            None,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
        )
        rotation_buffer_idx += 1

    times = bench_gpu_time(
        fn,
        enable_cupti=args.use_cupti,
        dry_run_iters=args.warmup,
        repeat_iters=args.iters,
        use_cuda_graph=args.use_cuda_graph,
    )
    torch.cuda.empty_cache()
    return float(np.median(times))


def main():
    parser = argparse.ArgumentParser(description="Benchmark GDN Prefill Kernel")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--cooling-time", type=float, default=0.1)
    parser.add_argument("--use-cupti", action="store_true")
    parser.add_argument("--use-cuda-graph", action="store_true")
    parser.add_argument(
        "--skip-cudnn",
        action="store_true",
        help="Do not measure the cuDNN backend column",
    )
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
        print(
            "Warning: FLA not installed (pip install flash-linear-attention). "
            "Benchmarking FlashInfer only."
        )

    cudnn_reason = None if args.skip_cudnn else _cudnn_rejection_reason(device)
    has_cudnn = not args.skip_cudnn and cudnn_reason is None
    if cudnn_reason is not None:
        print(f"Warning: cuDNN backend unavailable: {cudnn_reason}")

    print(f"\nGPU: {torch.cuda.get_device_name(0)} [{arch_label}]")
    print("Models: Qwen3.5 family (397B, 122B, 35B, 27B, 9B, 4B, 2B, 0.8B), d=128")
    print()
    fi_col = f"FI {arch_label}"
    header = (
        f"{'Heads':<15s}  {'Seqlens':<16s}  {'h_qk':>4s} {'h_v':>4s}"
        f"  {fi_col:>22s}  {'TFLOPS':>7s}"
    )
    if has_cudnn:
        header += f"  {'cuDNN':>10s}  {'FI/cuDNN':>9s}"
    if _has_fla:
        header += f"  {'FLA/Triton':>10s}  {'Speedup':>8s}"
    print(header)
    print("-" * len(header))

    order = ["FI"]
    if has_cudnn:
        order.append("cuDNN")
    if _has_fla:
        order.append("FLA")

    for h_qk, h_v, d, h_label in HEAD_CONFIGS:
        for endpoints, s_label in SEQ_CONFIGS:
            gc.collect()
            T = endpoints[-1]
            block_medians = {name: [] for name in order}
            for name in order + order[::-1]:
                if name == "FLA":
                    block_medians[name].append(bench_fla(args, endpoints, h_qk, h_v, d))
                else:
                    block_medians[name].append(
                        bench_fi(
                            args,
                            endpoints,
                            h_qk,
                            h_v,
                            d,
                            backend="cudnn" if name == "cuDNN" else "auto",
                        )
                    )
                time.sleep(args.cooling_time)
            fi_ms = float(np.median(block_medians["FI"]))
            tflops = _gdn_tflops(T, h_v, d, fi_ms)
            row = (
                f"{h_label:<15s}  {s_label:<16s}  {h_qk:>4d} {h_v:>4d}"
                f"  {fi_ms:>21.3f}ms  {tflops:>6.1f}"
            )
            if has_cudnn:
                cudnn_ms = float(np.median(block_medians["cuDNN"]))
                speedup = fi_ms / cudnn_ms
                marker = "+" if speedup > 1.0 else "-"
                row += f"  {cudnn_ms:>9.3f}ms  {speedup:>8.2f}x {marker}"
            if _has_fla:
                fla_ms = float(np.median(block_medians["FLA"]))
                speedup = fla_ms / fi_ms
                marker = "+" if speedup > 1.0 else "-"
                row += f"  {fla_ms:>9.3f}ms  {speedup:>7.2f}x {marker}"
            outer = block_medians[order[0]]
            drift = abs(outer[1] - outer[0]) / np.median(outer)
            if drift > DRIFT_THRESHOLD:
                row += f"  drift {drift * 100:.1f}%"
            print(row)
        print()


if __name__ == "__main__":
    main()
