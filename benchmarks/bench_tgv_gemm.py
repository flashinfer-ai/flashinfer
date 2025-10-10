#!/usr/bin/env python3
"""
Test bench for tgv_gemm_bf16_sm100() function.
Tests the autotuner integration with TGV BF16 GEMM kernels.
"""

import time
import csv
import torch
import torch.nn.functional as F

from flashinfer import tgv_gemm_sm100, autotune


def test_tgv_gemm_bf16_sm100_perf():
    """Test tgv_gemm_bf16_sm100 with different problem sizes."""
    print("\n=== Testing tgv_gemm_bf16_sm100 with different sizes ===")

    # Test different problem sizes
    test_cases = [
        (1, 7168, 2048, False, "deepseekv3, o_proj, tp=8"),
        (4, 7168, 2048, False, "deepseekv3, o_proj, tp=8"),
        (8, 7168, 2048, False, "deepseekv3, o_proj, tp=8"),
        (16, 7168, 2048, False, "deepseekv3, o_proj, tp=8"),
        (32, 7168, 2048, False, "deepseekv3, o_proj, tp=8"),
        (64, 7168, 2048, False, "deepseekv3, o_proj, tp=8"),
        (1, 3072, 1536, False, "deepseekv3, q_b_proj, tp=8"),
        (4, 3072, 1536, False, "deepseekv3, q_b_proj, tp=8"),
        (8, 3072, 1536, False, "deepseekv3, q_b_proj, tp=8"),
        (16, 3072, 1536, False, "deepseekv3, q_b_proj, tp=8"),
        (32, 3072, 1536, False, "deepseekv3, q_b_proj, tp=8"),
        (64, 3072, 1536, False, "deepseekv3, q_b_proj, tp=8"),
        (1, 1280, 2880, True, "gpt-oss-120b, qkv_proj, tp=4"),
        (4, 1280, 2880, True, "gpt-oss-120b, qkv_proj, tp=4"),
        (8, 1280, 2880, True, "gpt-oss-120b, qkv_proj, tp=4"),
        (16, 1280, 2880, True, "gpt-oss-120b, qkv_proj, tp=4"),
        (32, 1280, 2880, True, "gpt-oss-120b, qkv_proj, tp=4"),
        (64, 1280, 2880, True, "gpt-oss-120b, qkv_proj, tp=4"),
        (128, 1280, 2880, True, "gpt-oss-120b, qkv_proj, tp=4"),
        (1, 2880, 1024, True, "gpt-oss-120b, o_proj, tp=4"),
        (4, 2880, 1024, True, "gpt-oss-120b, o_proj, tp=4"),
        (8, 2880, 1024, True, "gpt-oss-120b, o_proj, tp=4"),
        (16, 2880, 1024, True, "gpt-oss-120b, o_proj, tp=4"),
        (32, 2880, 1024, True, "gpt-oss-120b, o_proj, tp=4"),
        (64, 2880, 1024, True, "gpt-oss-120b, o_proj, tp=4"),
        (128, 2880, 1024, True, "gpt-oss-120b, o_proj, tp=4"),
    ]

    # Prepare CSV output
    csv_filename = "bf16_tgv_gemm_benchmark_results.csv"
    csv_headers = [
        "M",
        "N",
        "K",
        "has_bias",
        "description",
        "cublas_time_ms",
        "tgv_time_ms",
        "pdl_time_ms",
        "tgv_speedup",
        "pdl_speedup",
    ]

    results = []

    for m, n, k, has_bias, description in test_cases:
        print(f"\n--- {description}: M={m}, N={n}, K={k}, has_bias={has_bias} ---")
        flops = m * n * k * 2 / 1e12
        # Create tensors
        A = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(n, k, device="cuda", dtype=torch.bfloat16).t()
        if has_bias:
            bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
        else:
            bias = None

        # Warmup cublas
        for _ in range(3):
            _ = F.linear(A, B.T, bias)

        torch.cuda.synchronize()

        cublas_graph = torch.cuda.CUDAGraph()

        # Start graph capture
        with torch.cuda.graph(cublas_graph):
            for _ in range(100):
                _ = F.linear(A, B.T, bias)

        # Warmup the graph
        for _ in range(3):
            cublas_graph.replay()

        torch.cuda.synchronize()

        # Benchmark using CUDA graph
        start_time = time.time()
        cublas_graph.replay()
        torch.cuda.synchronize()
        end_time = time.time()
        cublas_avg_time = (end_time - start_time) / 100
        print(
            f"CUBLAS average time: {cublas_avg_time * 1000:.6f} ms, {flops / cublas_avg_time:.3f} TFLOPS"
        )

        # Warmup
        with autotune(tune_mode=True):
            for _ in range(3):
                _ = tgv_gemm_sm100(A, B, bias)

        torch.cuda.synchronize()

        tgv_graph = torch.cuda.CUDAGraph()

        # Start graph capture
        with torch.cuda.graph(tgv_graph):
            for _ in range(100):
                _ = tgv_gemm_sm100(A, B, bias)

        # Warmup the graph
        tgv_graph.replay()

        torch.cuda.synchronize()

        # Benchmark using CUDA graph
        start_time = time.time()
        tgv_graph.replay()
        torch.cuda.synchronize()
        end_time = time.time()

        tgv_avg_time = (end_time - start_time) / 100
        print(
            f"TGV average time: {tgv_avg_time * 1000:.6f} ms, {flops / tgv_avg_time:.3f} TFLOPS, speedup: {cublas_avg_time / tgv_avg_time:.2f}x"
        )

        # Test with PDL
        print("\nTesting with PDL...")
        pdl_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(pdl_graph):
            for _ in range(100):
                _ = tgv_gemm_sm100(A, B, bias, pdl=True)

        # Warmup the graph
        pdl_graph.replay()

        torch.cuda.synchronize()

        # Benchmark using CUDA graph
        start_time = time.time()
        pdl_graph.replay()
        torch.cuda.synchronize()
        end_time = time.time()

        pdl_avg_time = (end_time - start_time) / 100
        print(
            f"PDL average time: {pdl_avg_time * 1000:.6f} ms, {flops / pdl_avg_time:.3f} TFLOPS, speedup: {cublas_avg_time / pdl_avg_time:.2f}x"
        )

        # Store results for CSV
        results.append(
            {
                "M": m,
                "N": n,
                "K": k,
                "has_bias": has_bias,
                "description": description,
                "cublas_time_ms": cublas_avg_time * 1000,
                "tgv_time_ms": tgv_avg_time * 1000,
                "pdl_time_ms": pdl_avg_time * 1000,
                "tgv_speedup": cublas_avg_time / tgv_avg_time,
                "pdl_speedup": cublas_avg_time / pdl_avg_time,
            }
        )

    # Write results to CSV
    print(f"\n=== Writing results to {csv_filename} ===")
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(results)

    print(f"Benchmark results saved to {csv_filename}")
    print(f"Total test cases: {len(results)}")


def test_tgv_gemm_bf16_sm100_correctness():
    """Test correctness of tgv_gemm_bf16_sm100 against reference implementation."""
    print("\n=== Testing correctness ===")

    # Create tensors
    m, n, k = 64, 2048, 1024
    A = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(n, k, device="cuda", dtype=torch.bfloat16).t()
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)

    # Reference computation
    reference = torch.matmul(A, B) + bias.unsqueeze(0)

    # Test with TGV runner
    out = tgv_gemm_sm100(A, B, bias)

    # Check correctness
    cos_sim = F.cosine_similarity(reference.reshape(-1), out.reshape(-1), dim=0)
    max_diff = torch.max(torch.abs(reference - out)).item()
    mean_diff = torch.mean(torch.abs(reference - out)).item()

    print(f"Cosine similarity: {cos_sim:.6f}")
    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")

    # Check if results are close enough
    if cos_sim > 0.99:
        print("✓ Correctness test PASSED")
    else:
        print("✗ Correctness test FAILED")


def main():
    """Run all tests for tgv_gemm_bf16_sm100."""
    print("Starting BF16 TGV GEMM SM100 Tests")
    print("=" * 50)

    try:
        # Run correctness test first
        test_tgv_gemm_bf16_sm100_correctness()

        # Test different problem sizes
        test_tgv_gemm_bf16_sm100_perf()

        print("\n" + "=" * 50)
        print("All BF16 TGV GEMM SM100 tests completed successfully!")

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
