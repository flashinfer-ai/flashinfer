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

Benchmark: Masked block-scaled BMM backend comparison (masked_scaled_bmm)

Compares FlashInfer's ``masked_scaled_bmm`` cuTile backend (PR #4020) against the
per-type SOTA backend the same masked-grouped-GEMM problem is served by:

  * nvfp4 -> ``grouped_gemm_nt_masked``      (cute-dsl, sm100/sm103)
  * mxfp8 -> ``group_gemm_fp8_nt_groupwise`` (trtllm, sm100+)

Layout: NT (trans_a=False, trans_b=True), out bf16. A: (Q, max_m, K) block-scaled,
B: (Q, N, K) block-scaled, per-group valid-row count in ``m_mask``.

The SOTA paths use independently generated scale factors in their own MMA layout
(the cuTile 5D TMA-descriptor scales are not layout-compatible), so SOTA is
**performance-only** and is NOT correctness-checked here -- exactly as in the
ocean validation suite. Only the cuTile output is verified against a dequantized
``torch.bmm`` reference on a small shape.

Usage:
    python bench_masked_scaled_bmm_backend_comparison.py
    python bench_masked_scaled_bmm_backend_comparison.py --providers cutile,sota --dtype mxfp8
    python bench_masked_scaled_bmm_backend_comparison.py --csv out.csv

Requirements:
    - SM100/SM103 (Blackwell); triton.tools.mxfp for data prep; matplotlib for heatmap
"""

import argparse
import csv as _csv
import random
from typing import Tuple

import numpy as np
import torch

import flashinfer
from flashinfer.gemm import masked_scaled_bmm
from flashinfer.testing.utils import bench_gpu_time

try:
    from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor

    HAS_MXFP = True
except ImportError:
    HAS_MXFP = False

ALL_PROVIDERS = ["cutile", "sota"]


def get_cc() -> int:
    """Compute capability of the current device as a two-digit int (e.g. 100 for sm100)."""
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def create_masked_m(num_groups, expected_m_per_group, max_m):
    """Random per-group valid-row counts (multiple of 128), bounded by max_m. Deterministic under seed."""
    masked_m = torch.empty((num_groups,), dtype=torch.int32, device="cuda")
    for j in range(num_groups):
        masked_m[j] = (
            int(expected_m_per_group * random.uniform(0.7, 1.3) + 127) // 128 * 128
        )
    masked_m.clamp_(max=max_m)
    return masked_m


def initialize_block_scaled(
    num_groups,
    max_m,
    expected_m_per_group,
    N,
    K,
    block_scale_type,
    out_dtype=torch.bfloat16,
):
    """Build packed cuTile inputs (a, b, a_scale, b_scale, m_mask) mirroring the test data-prep."""
    VEC_SIZE = 16 if block_scale_type == "nvfp4" else 32
    Q = num_groups
    M = max_m
    m_mask = create_masked_m(num_groups, expected_m_per_group, max_m)

    device = "cuda"
    a_ref = MXFP4Tensor(size=(Q, M, K), device=device).random()
    b_ref = MXFP4Tensor(size=(Q, N, K), device=device).random()
    if block_scale_type in ["mxfp8", "mixed"]:
        a = a_ref.to(torch.float32).to(torch.float8_e4m3fn)
    else:
        a = a_ref.to_packed_tensor(dim=2)
    if block_scale_type == "mxfp8":
        b = b_ref.to(torch.float32).to(torch.float8_e4m3fn)
    else:
        b = b_ref.to_packed_tensor(dim=2)

    a_scale_shape = [Q, M // 128, K // VEC_SIZE // 4, 32, 16]
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
    a_scale = a_scale.reshape(Q, a_scale_shape[1], a_scale.shape[2], 2, 256)
    b_scale = b_scale.reshape(Q, b_scale_shape[1], b_scale.shape[2], 2, 256)
    return a, b, a_scale, b_scale, m_mask


def _make_sota_nvfp4(a, b, m_mask, num_groups, max_m, N, K, out_dtype):
    """Independent-scale inputs for the cute-dsl grouped_gemm_nt_masked (perf-only)."""
    Q, M = num_groups, max_m
    sf_vec_size = 16
    a_sota = a.permute(1, 2, 0).contiguous()  # (Q,M,K/2) -> (M,K/2,Q)
    b_sota = b.permute(1, 2, 0).contiguous()  # (Q,N,K/2) -> (N,K/2,Q)
    c_sota = torch.empty(Q, M, N, device="cuda", dtype=out_dtype)
    sf_k = (K + sf_vec_size - 1) // sf_vec_size
    m_tiles = (M + 127) // 128
    n_tiles = (N + 127) // 128
    k_tiles = (sf_k + 3) // 4
    sfa = (
        torch.randn(Q, m_tiles, k_tiles, 32, 4, 4, device="cuda")
        .to(torch.float8_e4m3fn)
        .permute(3, 4, 1, 5, 2, 0)
    )
    sfb = (
        torch.randn(Q, n_tiles, k_tiles, 32, 4, 4, device="cuda")
        .to(torch.float8_e4m3fn)
        .permute(3, 4, 1, 5, 2, 0)
    )
    return a_sota, b_sota, c_sota, sfa, sfb, sf_vec_size


def _make_sota_mxfp8(a, b, m_mask, num_groups, N, K, out_dtype):
    """Independent-scale inputs for the trtllm group_gemm_fp8_nt_groupwise (perf-only)."""
    Q = num_groups
    block_size = 128
    k_tiles = K // block_size
    n_tiles = N // block_size
    m_counts = m_mask.tolist()
    total_m_actual = sum(m_counts)
    m_indptr = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            m_mask.cumsum(0).to(torch.int32),
        ]
    )
    a_scale = (
        torch.rand(total_m_actual, k_tiles, dtype=torch.float32, device="cuda") * 1e-2
    )
    b_scale = torch.rand(Q, n_tiles, k_tiles, dtype=torch.float32, device="cuda") * 1e-2
    a_flat = torch.cat([a[i, : m_counts[i], :] for i in range(Q)], dim=0)
    out = torch.empty(total_m_actual, N, dtype=out_dtype, device="cuda")
    return a_flat, a_scale, b_scale, m_indptr, out, block_size


def make_call(provider, block_scale_type, num_groups, max_m, exp_m, N, K, out_dtype):
    """Return (callable, effective_flops) for one provider on one shape, or (None, 0) if unavailable."""
    torch.manual_seed(0)
    random.seed(0)
    a, b, a_scale, b_scale, m_mask = initialize_block_scaled(
        num_groups, max_m, exp_m, N, K, block_scale_type, out_dtype
    )
    flops = 2.0 * float(m_mask.sum().item()) * N * K  # sum over valid rows

    if provider == "cutile":

        def fn():
            return masked_scaled_bmm(
                a,
                b,
                a_scale,
                b_scale,
                m_mask,
                block_scale_type,
                max_m_device=None,
                transpose_a=False,
                transpose_b=True,
                out_dtype=out_dtype,
                backend="cutile",
            )

        return fn, flops

    if provider == "sota":
        if block_scale_type == "nvfp4":
            from flashinfer.gemm import grouped_gemm_nt_masked

            a_s, b_s, c_s, sfa, sfb, vec = _make_sota_nvfp4(
                a, b, m_mask, num_groups, max_m, N, K, out_dtype
            )
            c_dtype = "bfloat16" if out_dtype == torch.bfloat16 else "float16"

            def fn():
                return grouped_gemm_nt_masked(
                    (a_s, sfa),
                    (b_s, sfb),
                    c_s,
                    m_mask,
                    ab_dtype="float4_e2m1fn",
                    sf_dtype="float8_e4m3fn",
                    c_dtype=c_dtype,
                    sf_vec_size=vec,
                )

            return fn, flops
        else:  # mxfp8
            from flashinfer.gemm import group_gemm_fp8_nt_groupwise

            a_flat, a_sc, b_sc, m_indptr, out, bs = _make_sota_mxfp8(
                a, b, m_mask, num_groups, N, K, out_dtype
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


def bench_one(provider, block_scale_type, num_groups, max_m, exp_m, N, K, out_dtype):
    """Median latency (ms) and TFLOP/s; NaN if the provider errors on the shape."""
    try:
        fn, flops = make_call(
            provider, block_scale_type, num_groups, max_m, exp_m, N, K, out_dtype
        )
        if fn is None:
            return float("nan"), float("nan")
        fn()  # warmup / autotune
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
    """Dequantized torch.bmm reference check for the cuTile backend on a small shape."""
    num_groups, max_m, exp_m, N, K = 2, 512, 64, 256, 256
    VEC_SIZE = 16 if block_scale_type == "nvfp4" else 32
    Q, M = num_groups, max_m
    torch.manual_seed(0)
    random.seed(0)
    device = "cuda"
    m_mask = create_masked_m(num_groups, exp_m, max_m)
    a_ref = MXFP4Tensor(size=(Q, M, K), device=device).random()
    b_ref = MXFP4Tensor(size=(Q, N, K), device=device).random()
    if block_scale_type in ["mxfp8", "mixed"]:
        a_ref = a_ref.to(torch.float32)
        a = a_ref.to(torch.float8_e4m3fn)
    else:
        a = a_ref.to_packed_tensor(dim=2)
    if block_scale_type == "mxfp8":
        b_ref = b_ref.to(torch.float32)
        b = b_ref.to(torch.float8_e4m3fn)
    else:
        b = b_ref.to_packed_tensor(dim=2)
    b_ref = b_ref.to(torch.float32)
    b_ref = torch.transpose(b_ref, 1, 2)
    a_scale_shape = [Q, M // 128, K // VEC_SIZE // 4, 32, 16]
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
    a_scale = a_scale.reshape(Q, a_scale_shape[1], a_scale.shape[2], 2, 256)
    b_scale = b_scale.reshape(Q, b_scale_shape[1], b_scale.shape[2], 2, 256)

    a_scale_ref = a_scale_ref.to(torch.float32)
    b_scale_ref = b_scale_ref.to(torch.float32)

    def unpack_scale(packed):
        packed = packed.reshape(*packed.shape[:-2], 32, 4, 4)
        nq, nm, nk, _, _, _ = packed.shape
        return (
            packed.permute(0, 1, 4, 3, 2, 5).reshape(nq, nm * 128, nk * 4).contiguous()
        )

    a_scale_ref = unpack_scale(a_scale_ref).repeat_interleave(VEC_SIZE, dim=2)[
        :Q, :M, :K
    ]
    b_scale_ref = (
        unpack_scale(b_scale_ref)
        .repeat_interleave(VEC_SIZE, dim=2)
        .permute(0, 2, 1)
        .contiguous()[:Q, :K, :N]
    )
    a_ref_float = a_ref.to(torch.float32)
    for i in range(Q):
        a_ref_float[i, m_mask[i] :, :] = 0
    reference = torch.bmm(a_ref_float * a_scale_ref, b_ref * b_scale_ref).to(out_dtype)

    c = masked_scaled_bmm(
        a,
        b,
        a_scale,
        b_scale,
        m_mask,
        block_scale_type,
        max_m_device=None,
        transpose_a=False,
        transpose_b=True,
        out_dtype=out_dtype,
        backend="cutile",
    )
    for i in range(num_groups):
        c[i, m_mask[i] :, :] = 0
    try:
        torch.testing.assert_close(reference, c, atol=1e-2, rtol=1e-2)
        return True, 0.0
    except AssertionError:
        cos = torch.nn.functional.cosine_similarity(
            reference.float().reshape(1, -1), c.float().reshape(1, -1)
        ).item()
        return cos > 0.99, cos


def default_workloads():
    """(num_groups, max_m, expected_m_per_group, n, k) -- ocean DeepGEMM/GB200 named cases."""
    cases = [(6, 1024), (6, 512), (1, 1024), (2, 512), (4, 256)]
    out = []
    for ng, em in cases:
        for n, k in ((4096, 7168), (7168, 2048)):
            out.append((ng, 4096, em, n, k))
    return out


def main():
    parser = argparse.ArgumentParser(description="Benchmark masked_scaled_bmm backends")
    parser.add_argument("--providers", type=str, default=",".join(ALL_PROVIDERS))
    parser.add_argument(
        "--dtype",
        type=str,
        default="mxfp8,nvfp4",
        help="comma list of block_scale_type: mxfp8,nvfp4",
    )
    parser.add_argument("--baseline", type=str, default="sota")
    parser.add_argument(
        "--output-prefix", type=str, default="masked_scaled_bmm_backend_comparison"
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
        print(f"\n{'#' * 78}\n# block_scale_type = {bst}\n{'#' * 78}")
        print("Correctness (cuTile vs dequant torch.bmm, 2x512x256x256):")
        ok, cos = verify_cutile(bst, out_dtype)
        print(
            f"  cutile: {'OK' if ok else 'FAIL'}" + (f" cos={cos:.4f}" if cos else "")
        )
        print("  sota  : perf-only (independent scales; not correctness-comparable)")

        results = {p: {} for p in providers}
        tflops = {p: {} for p in providers}
        header = f"{'Q':>4} {'max_m':>6} {'em':>5} {'n':>6} {'k':>6} |"
        for p in providers:
            header += f" {p + '_ms':>12} {p + '_TF/s':>10}"
        print("\n" + header)
        print("-" * len(header))
        for ng, mm, em, n, k in workloads:
            row = f"{ng:>4} {mm:>6} {em:>5} {n:>6} {k:>6} |"
            for p in providers:
                ms, tf = bench_one(p, bst, ng, mm, em, n, k, out_dtype)
                results[p][(ng, mm, em, n, k)] = ms
                tflops[p][(ng, mm, em, n, k)] = tf
                if ms == ms:
                    row += f" {ms:>12.4f} {tf:>10.1f}"
                else:
                    row += f" {'--':>12} {'--':>10}"
            print(row)
        all_results[bst] = (results, tflops)

        # speedup summary
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
                    "max_m",
                    "expected_m_per_group",
                    "n",
                    "k",
                    "median_ms",
                    "tflops",
                ]
            )
            for bst, (results, tflops) in all_results.items():
                for p in providers:
                    for (ng, mm, em, n, k), ms in sorted(results[p].items()):
                        tf = tflops[p][(ng, mm, em, n, k)]
                        w.writerow([bst, p, ng, mm, em, n, k, f"{ms:.6f}", f"{tf:.2f}"])
        print(f"\nWrote {args.csv}")

    print("\n" + "=" * 78 + "\nBENCHMARK COMPLETE\n" + "=" * 78)


if __name__ == "__main__":
    main()
