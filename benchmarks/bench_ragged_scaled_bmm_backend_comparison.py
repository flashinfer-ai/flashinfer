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

Benchmark: Ragged block-scaled BMM backend comparison (ragged_scaled_bmm)

Compares FlashInfer's ``ragged_scaled_bmm`` cuTile backend (PR #4020) against SOTA.
A is a ragged stack ``(total_m, K)`` split into per-group segments by
``segment_offsets``; B is batched ``(Q, N, K)``; layout NT, out bf16.

SOTA path (mxfp8 only): ``group_gemm_fp8_nt_groupwise`` (trtllm) with the ragged
segment offsets used directly as ``m_indptr``. It is served with independently
generated scale factors in its own layout, so it is **performance-only** and NOT
correctness-checked (as in the ocean suite). There is no SOTA ragged path for
nvfp4, so nvfp4 is cuTile-only here.

Only the cuTile output is verified against the dequantized per-segment torch.mm
reference (small shape).

Usage:
    python bench_ragged_scaled_bmm_backend_comparison.py
    python bench_ragged_scaled_bmm_backend_comparison.py --dtype mxfp8 --csv out.csv

Requirements:
    - SM100/SM103 (Blackwell); triton.tools.mxfp for data prep
"""

import argparse
import csv as _csv
import random
from typing import Tuple

import numpy as np
import torch

import flashinfer
from flashinfer.gemm import ragged_scaled_bmm
from flashinfer.testing.utils import bench_gpu_time

try:
    from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor

    HAS_MXFP = True
except ImportError:
    HAS_MXFP = False

ALL_PROVIDERS = ["cutile", "sota"]


def get_cc() -> int:
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def create_ragged_m_segments(num_groups, m, ELEM_PER_BYTE_A):
    """Non-even, 128-aligned M segments summing to num_groups*m. Deterministic under seed."""
    total_m = num_groups * m
    num_items = 16 * ELEM_PER_BYTE_A
    alignment = 128
    sizes = []
    for _ in range(num_groups - 1):
        s = int(m * random.uniform(0.5, 1.5))
        s = (s // num_items) * num_items
        s = (s // alignment) * alignment
        sizes.append(s)
    remaining = total_m - sum(sizes)
    assert remaining > 0 and remaining % num_items == 0 and remaining % alignment == 0
    sizes.append(remaining)
    segment_offsets = torch.zeros(num_groups + 1, dtype=torch.int32, device="cuda")
    for i in range(num_groups):
        segment_offsets[i + 1] = segment_offsets[i] + sizes[i]
    return max(sizes), segment_offsets


def initialize_block_scaled(num_groups, M, N, K, block_scale_type):
    """Build packed cuTile ragged inputs (a, b, a_scale, b_scale, segment_offsets, max_m)."""
    VEC_SIZE = 16 if block_scale_type == "nvfp4" else 32
    ELEM_PER_BYTE_A = 2 if "fp4" in block_scale_type else 1
    Q = num_groups
    max_m, segment_offsets = create_ragged_m_segments(num_groups, M, ELEM_PER_BYTE_A)
    total_m = segment_offsets[-1].item()

    device = "cuda"
    a_ref = MXFP4Tensor(size=(total_m, K), device=device).random()
    b_ref = MXFP4Tensor(size=(Q, N, K), device=device).random()
    if block_scale_type in ["mxfp8", "mixed"]:
        a = a_ref.to(torch.float32).to(torch.float8_e4m3fn)
    else:
        a = a_ref.to_packed_tensor(dim=1)
    if block_scale_type == "mxfp8":
        b = b_ref.to(torch.float32).to(torch.float8_e4m3fn)
    else:
        b = b_ref.to_packed_tensor(dim=2)

    a_scale_shape = [total_m // 128, K // VEC_SIZE // 4, 32, 16]
    b_scale_shape = [Q, N // 128, K // VEC_SIZE // 4, 32, 16]
    eps = 1e-8
    a_scale = torch.rand(a_scale_shape, device=device) + eps
    b_scale = torch.rand(b_scale_shape, device=device) + eps
    if block_scale_type == "nvfp4":
        a_scale = a_scale.to(torch.float8_e4m3fn)
        b_scale = b_scale.to(torch.float8_e4m3fn)
    else:
        a_scale = MXScaleTensor(a_scale).data
        b_scale = MXScaleTensor(b_scale).data
    a_scale = a_scale.reshape(a_scale_shape[0], a_scale.shape[1], 2, 256)
    b_scale = b_scale.reshape(Q, b_scale_shape[1], b_scale.shape[2], 2, 256)
    return a, b, a_scale, b_scale, segment_offsets, max_m, total_m


def _make_sota_mxfp8(a, b, segment_offsets, num_groups, total_m, N, K, out_dtype):
    """Independent-scale inputs for the trtllm group_gemm_fp8_nt_groupwise (perf-only)."""
    Q = num_groups
    block_size = 128
    k_tiles = K // block_size
    n_tiles = N // block_size
    m_indptr = segment_offsets.clone().to(torch.int32)
    a_scale = torch.rand(total_m, k_tiles, dtype=torch.float32, device="cuda") * 1e-2
    b_scale = torch.rand(Q, n_tiles, k_tiles, dtype=torch.float32, device="cuda") * 1e-2
    out = torch.empty(total_m, N, dtype=out_dtype, device="cuda")
    return a, a_scale, b_scale, m_indptr, out, block_size


def make_call(provider, block_scale_type, num_groups, M, N, K, out_dtype):
    torch.manual_seed(0)
    random.seed(0)
    a, b, a_scale, b_scale, seg_off, max_m, total_m = initialize_block_scaled(
        num_groups, M, N, K, block_scale_type
    )
    flops = 2.0 * float(total_m) * N * K

    if provider == "cutile":

        def fn():
            return ragged_scaled_bmm(
                a,
                b,
                a_scale,
                b_scale,
                seg_off,
                max_m,
                block_scale_type,
                transpose_a=False,
                transpose_b=True,
                backend="cutile",
            )

        return fn, flops

    if provider == "sota":
        if block_scale_type != "mxfp8":
            return None, 0.0  # no ragged SOTA path for nvfp4
        from flashinfer.gemm import group_gemm_fp8_nt_groupwise

        a_flat, a_sc, b_sc, m_indptr, out, bs = _make_sota_mxfp8(
            a, b, seg_off, num_groups, total_m, N, K, out_dtype
        )

        def fn():
            return group_gemm_fp8_nt_groupwise(
                a_flat,
                b,
                a_sc,
                b_sc,
                m_indptr,
                scale_granularity_mnk=(1, bs, bs),
                scale_major_mode="K",
                out=out,
                out_dtype=out_dtype,
            )

        return fn, flops
    return None, 0.0


def bench_one(provider, block_scale_type, num_groups, M, N, K, out_dtype):
    try:
        fn, flops = make_call(
            provider, block_scale_type, num_groups, M, N, K, out_dtype
        )
        if fn is None:
            return float("nan"), float("nan")
        fn()
        torch.cuda.synchronize()
        times = bench_gpu_time(
            fn=fn,
            enable_cupti=False,
            dry_run_iters=5,
            repeat_iters=30,
            cold_l2_cache=True,
            use_cuda_graph=False,
        )
        ms = float(np.median(times))
        tf = flops / (ms * 1e-3) / 1e12 if ms > 0 else float("nan")
        return ms, tf
    except Exception as e:
        print(
            f"    [{provider}/{block_scale_type}] {type(e).__name__}: {str(e).splitlines()[0][:120]}"
        )
        return float("nan"), float("nan")


def verify_cutile(block_scale_type, out_dtype=torch.bfloat16) -> Tuple[bool, float]:
    """Dequantized per-segment torch.mm reference check for the cuTile backend (small shape)."""
    num_groups, M, N, K = 3, 512, 2048, 1024
    VEC_SIZE = 16 if block_scale_type == "nvfp4" else 32
    torch.manual_seed(0)
    random.seed(0)
    device = "cuda"
    ELEM_PER_BYTE_A = 2 if "fp4" in block_scale_type else 1
    max_m, seg_off = create_ragged_m_segments(num_groups, M, ELEM_PER_BYTE_A)
    total_m = seg_off[-1].item()
    Q = num_groups
    a_ref = MXFP4Tensor(size=(total_m, K), device=device).random()
    b_ref = MXFP4Tensor(size=(Q, N, K), device=device).random()
    if block_scale_type in ["mxfp8", "mixed"]:
        a_ref = a_ref.to(torch.float32)
        a = a_ref.to(torch.float8_e4m3fn)
    else:
        a = a_ref.to_packed_tensor(dim=1)
    if block_scale_type == "mxfp8":
        b_ref = b_ref.to(torch.float32)
        b = b_ref.to(torch.float8_e4m3fn)
    else:
        b = b_ref.to_packed_tensor(dim=2)
    b_ref = b_ref.to(torch.float32)
    b_ref = torch.transpose(b_ref, 1, 2)

    a_scale_shape = [total_m // 128, K // VEC_SIZE // 4, 32, 16]
    b_scale_shape = [Q, N // 128, K // VEC_SIZE // 4, 32, 16]
    a_scale = torch.rand(a_scale_shape, device=device) + 1e-8
    b_scale = torch.rand(b_scale_shape, device=device) + 1e-8
    if block_scale_type == "nvfp4":
        a_scale = a_scale.to(torch.float8_e4m3fn)
        b_scale = b_scale.to(torch.float8_e4m3fn)
        a_scale_ref, b_scale_ref = a_scale, b_scale
    else:
        a_scale_ref = MXScaleTensor(a_scale)
        b_scale_ref = MXScaleTensor(b_scale)
        a_scale = a_scale_ref.data
        b_scale = b_scale_ref.data
    a_scale = a_scale.reshape(a_scale_shape[0], a_scale.shape[1], 2, 256)
    b_scale = b_scale.reshape(Q, b_scale_shape[1], b_scale.shape[2], 2, 256)

    packed_a = (
        a_scale_ref.to(torch.float32)
        .reshape(a_scale_shape[0], a_scale.shape[1], 32, 4, 4)
        .permute(0, 3, 2, 1, 4)
        .reshape(a_scale_shape[0] * 128, a_scale.shape[1] * 4)
        .contiguous()
    )
    unpacked_a = packed_a.repeat_interleave(VEC_SIZE, dim=1).contiguous()[:total_m, :K]
    packed_b = (
        b_scale_ref.to(torch.float32)
        .reshape(Q, b_scale_shape[1], b_scale.shape[2], 32, 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .reshape(Q, b_scale_shape[1] * 128, b_scale.shape[2] * 4)
        .contiguous()
    )
    unpacked_b = (
        packed_b.repeat_interleave(VEC_SIZE, dim=2)
        .permute(0, 2, 1)
        .contiguous()[:Q, :K, :N]
    )
    a_ref_float = a_ref.to(torch.float32)

    a_deq = a_ref_float * unpacked_a
    b_deq = b_ref * unpacked_b
    ref = torch.zeros((total_m, N), device=device, dtype=torch.float32)
    for q in range(Q):
        s, e = seg_off[q].item(), seg_off[q + 1].item()
        ref[s:e, :] = torch.mm(a_deq[s:e, :], b_deq[q, :, :])

    c = ragged_scaled_bmm(
        a,
        b,
        a_scale,
        b_scale,
        seg_off,
        max_m,
        block_scale_type,
        transpose_a=False,
        transpose_b=True,
        backend="cutile",
    )
    try:
        torch.testing.assert_close(ref, c.to(torch.float32), atol=1e-2, rtol=1e-2)
        return True, 0.0
    except AssertionError:
        cos = torch.nn.functional.cosine_similarity(
            ref.reshape(1, -1), c.float().reshape(1, -1)
        ).item()
        return cos > 0.99, cos


def default_workloads():
    """(num_groups, m, n, k) -- ragged grouped-GEMM shapes."""
    return [
        (3, 512, 2048, 1024),
        (4, 512, 4096, 7168),
        (8, 512, 7168, 2048),
        (2, 1024, 4096, 7168),
        (4, 1024, 7168, 2048),
    ]


def main():
    parser = argparse.ArgumentParser(description="Benchmark ragged_scaled_bmm backends")
    parser.add_argument("--providers", type=str, default=",".join(ALL_PROVIDERS))
    parser.add_argument("--dtype", type=str, default="mxfp8,nvfp4")
    parser.add_argument("--baseline", type=str, default="sota")
    parser.add_argument(
        "--output-prefix", type=str, default="ragged_scaled_bmm_backend_comparison"
    )
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()

    if not HAS_MXFP:
        print("triton.tools.mxfp unavailable; cannot prepare FP4/FP8 inputs. Abort.")
        return
    cc = get_cc()
    print(f"GPU Compute Capability: SM{cc}")
    print(f"flashinfer from: {flashinfer.__file__}")
    providers = [
        p.strip() for p in args.providers.split(",") if p.strip() in ALL_PROVIDERS
    ]
    dtypes = [d.strip() for d in args.dtype.split(",") if d.strip()]
    workloads = default_workloads()
    out_dtype = torch.bfloat16

    all_results = {}
    for bst in dtypes:
        print(f"\n{'#' * 74}\n# block_scale_type = {bst}\n{'#' * 74}")
        print("Correctness (cuTile vs dequant per-segment torch.mm, 3x512x2048x1024):")
        ok, cos = verify_cutile(bst, out_dtype)
        print(
            f"  cutile: {'OK' if ok else 'FAIL'}" + (f" cos={cos:.4f}" if cos else "")
        )
        if bst == "mxfp8":
            print(
                "  sota  : perf-only (independent scales; not correctness-comparable)"
            )
        else:
            print("  sota  : n/a (no ragged SOTA path for nvfp4)")

        results = {p: {} for p in providers}
        tflops = {p: {} for p in providers}
        header = f"{'Q':>4} {'m':>6} {'n':>6} {'k':>6} |"
        for p in providers:
            header += f" {p + '_ms':>12} {p + '_TF/s':>10}"
        print("\n" + header)
        print("-" * len(header))
        for ng, m, n, k in workloads:
            row = f"{ng:>4} {m:>6} {n:>6} {k:>6} |"
            for p in providers:
                ms, tf = bench_one(p, bst, ng, m, n, k, out_dtype)
                results[p][(ng, m, n, k)] = ms
                tflops[p][(ng, m, n, k)] = tf
                if ms == ms:
                    row += f" {ms:>12.4f} {tf:>10.1f}"
                else:
                    row += f" {'--':>12} {'--':>10}"
            print(row)
        all_results[bst] = (results, tflops)

        baseline = args.baseline if args.baseline in providers else providers[0]
        others = [p for p in providers if p != baseline]
        if others:
            print(
                f"\nSpeedup vs {baseline} ({baseline}_ms / provider_ms; >1 = provider faster):"
            )
            geo = {p: [] for p in others}
            for key in workloads:
                bt = results[baseline].get(key, float("nan"))
                for p in others:
                    pt = results[p].get(key, float("nan"))
                    if pt == pt and bt == bt and pt > 0:
                        geo[p].append(bt / pt)
            for p in others:
                if geo[p]:
                    print(
                        f"  {p}: geomean {np.exp(np.mean(np.log(geo[p]))):.2f}x "
                        f"(min {min(geo[p]):.2f}, max {max(geo[p]):.2f}, "
                        f"{sum(1 for r in geo[p] if r > 1)}/{len(geo[p])} faster)"
                    )

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(
                [
                    "block_scale_type",
                    "provider",
                    "num_groups",
                    "m",
                    "n",
                    "k",
                    "median_ms",
                    "tflops",
                ]
            )
            for bst, (results, tflops) in all_results.items():
                for p in providers:
                    for (ng, m, n, k), ms in sorted(results[p].items()):
                        tf = tflops[p][(ng, m, n, k)]
                        w.writerow([bst, p, ng, m, n, k, f"{ms:.6f}", f"{tf:.2f}"])
        print(f"\nWrote {args.csv}")

    print("\n" + "=" * 74 + "\nBENCHMARK COMPLETE\n" + "=" * 74)


if __name__ == "__main__":
    main()
