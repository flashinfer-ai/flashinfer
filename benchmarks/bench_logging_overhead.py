#!/usr/bin/env python3
"""
Benchmark script to measure the overhead of API logging at different levels.

This script creates decorated and undecorated versions of a test function
(torch.matmul) and compares their performance to accurately measure logging overhead.

Usage:
    # Set the logging level before running
    export FLASHINFER_LOGLEVEL=3
    python bench_logging_overhead.py

    # Or run with different levels
    FLASHINFER_LOGLEVEL=0 python bench_logging_overhead.py
    FLASHINFER_LOGLEVEL=1 python bench_logging_overhead.py
    FLASHINFER_LOGLEVEL=3 python bench_logging_overhead.py
    FLASHINFER_LOGLEVEL=5 python bench_logging_overhead.py

    # Or use the helper script to run all levels
    bash benchmark_all_levels.sh
"""

import os
import sys
import time
import torch
import numpy as np
from typing import List, Tuple

# Get logging level BEFORE importing flashinfer
LOGGING_LEVEL = int(os.environ.get("FLASHINFER_LOGLEVEL", "0"))
LOG_DEST = os.environ.get("FLASHINFER_LOGDEST", "/tmp/flashinfer_benchmark_log.txt")

# Import the decorator
from flashinfer.api_logging import flashinfer_api


# Create two versions of a test function:
# 1. Undecorated (baseline)
# 2. Decorated (with logging)
def test_matmul_undecorated(A, B):
    return torch.matmul(A, B)


@flashinfer_api
def test_matmul_decorated(A, B):
    return torch.matmul(A, B)


class BenchmarkResults:
    """Store and display benchmark results."""

    def __init__(self):
        self.undecorated_times = []
        self.decorated_times = []

    def set_undecorated(self, times: List[float]):
        """Set benchmark results for undecorated function."""
        self.undecorated_times = times

    def set_decorated(self, times: List[float]):
        """Set benchmark results for decorated function."""
        self.decorated_times = times

    def print_summary(self, logging_level: int):
        """Print a summary of benchmark results."""
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS")
        print("=" * 80)

        undecorated_mean = np.mean(self.undecorated_times)
        undecorated_std = np.std(self.undecorated_times)

        decorated_mean = np.mean(self.decorated_times)
        decorated_std = np.std(self.decorated_times)

        overhead_abs = (decorated_mean - undecorated_mean) * 1000  # ms
        overhead_pct = (
            ((decorated_mean - undecorated_mean) / undecorated_mean * 100)
            if undecorated_mean > 0
            else 0
        )

        print(
            f"\n{'Version':<20} {'Mean (ms)':<12} {'Std (ms)':<12} {'Median (ms)':<12}"
        )
        print("-" * 80)
        print(
            f"{'Undecorated':<20} {undecorated_mean * 1000:<12.4f} {undecorated_std * 1000:<12.4f} {np.median(self.undecorated_times) * 1000:<12.4f}"
        )
        print(
            f"{'Decorated':<20} {decorated_mean * 1000:<12.4f} {decorated_std * 1000:<12.4f} {np.median(self.decorated_times) * 1000:<12.4f}"
        )

        print("\n" + "=" * 80)
        print("OVERHEAD ANALYSIS")
        print("=" * 80)
        print(f"\nLogging Level: {logging_level}")
        print(f"Absolute overhead: {overhead_abs:.4f} ms")
        print(f"Relative overhead: {overhead_pct:.2f}%")

        print("\n" + "=" * 80)
        print("DETAILED STATISTICS")
        print("=" * 80)

        print("\nUndecorated (baseline):")
        print(f"  Mean:   {undecorated_mean * 1000:.4f} ms")
        print(f"  Median: {np.median(self.undecorated_times) * 1000:.4f} ms")
        print(f"  Std:    {undecorated_std * 1000:.4f} ms")
        print(f"  Min:    {np.min(self.undecorated_times) * 1000:.4f} ms")
        print(f"  Max:    {np.max(self.undecorated_times) * 1000:.4f} ms")

        print("\nDecorated (with logging):")
        print(f"  Mean:   {decorated_mean * 1000:.4f} ms")
        print(f"  Median: {np.median(self.decorated_times) * 1000:.4f} ms")
        print(f"  Std:    {decorated_std * 1000:.4f} ms")
        print(f"  Min:    {np.min(self.decorated_times) * 1000:.4f} ms")
        print(f"  Max:    {np.max(self.decorated_times) * 1000:.4f} ms")


def setup_test_inputs(
    batch_size: int = 32,
    m: int = 512,
    n: int = 512,
    k: int = 512,
    device: str = "cuda:0",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Set up test inputs for matmul.

    Parameters
    ----------
    batch_size : int
        Batch size for the matrix multiplication
    m, n, k : int
        Matrix dimensions
    device : str
        Device to use

    Returns
    -------
    A, B : torch.Tensor
        Input tensors for matrix multiplication
    """
    # Create random tensors
    A = torch.randn(batch_size, m, k, dtype=torch.float16, device=device)
    B = torch.randn(batch_size, k, n, dtype=torch.float16, device=device)

    return A, B


def warmup(func, A, B, num_warmup: int = 10):
    """Warmup the GPU and JIT compilation."""
    for _ in range(num_warmup):
        _ = func(A, B)
    torch.cuda.synchronize()


def benchmark_function(
    func, func_name: str, A, B, num_iterations: int = 100
) -> List[float]:
    """
    Benchmark a specific function.

    Parameters
    ----------
    func : callable
        Function to benchmark
    func_name : str
        Name of the function (for display)
    A, B : torch.Tensor
        Input tensors for matrix multiplication
    num_iterations : int
        Number of iterations to run

    Returns
    -------
    List[float]
        List of execution times in seconds
    """
    print(f"\nBenchmarking: {func_name}")
    print(f"  Running {num_iterations} iterations...")

    times = []

    for _ in range(num_iterations):
        # Synchronize before timing
        torch.cuda.synchronize()

        # Time the execution
        start = time.perf_counter()
        _ = func(A, B)
        torch.cuda.synchronize()
        end = time.perf_counter()

        elapsed = end - start
        times.append(elapsed)

    print(f"  Complete. Mean time: {np.mean(times) * 1000:.4f} ms")

    return times


def main():
    """Main benchmark function."""
    print("=" * 80)
    print("FlashInfer API Logging Overhead Benchmark")
    print("=" * 80)

    # Display logging configuration
    print("\nLogging Configuration:")
    print(f"  FLASHINFER_LOGLEVEL = {LOGGING_LEVEL}")
    print(f"  FLASHINFER_LOGDEST = {LOG_DEST}")

    # Get level name
    level_names = {
        0: "No logging (zero-overhead)",
        1: "Function name only",
        3: "Name + inputs/outputs + metadata",
        5: "Name + inputs/outputs + metadata + statistics",
    }
    print(f"  Level description: {level_names.get(LOGGING_LEVEL, 'Unknown')}")

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("\nError: CUDA is not available. This benchmark requires a CUDA device.")
        exit(1)

    device = "cuda:0"
    print(f"\nDevice: {device}")
    print(f"Device Name: {torch.cuda.get_device_name(device)}")

    # Setup test inputs
    print("\nSetting up test inputs...")
    batch_size = 32
    m, n, k = 128, 128, 128
    print(f"  Batch size: {batch_size}")
    print(f"  Matrix dimensions: [{batch_size}, {m}, {k}] @ [{batch_size}, {k}, {n}]")

    A, B = setup_test_inputs(batch_size, m, n, k, device)

    # Benchmark parameters
    num_iterations = 100
    print("\nBenchmark parameters:")
    print(f"  Iterations: {num_iterations}")
    print("  Warmup iterations: 10")

    # Clear log file before starting
    if os.path.exists(LOG_DEST):
        os.remove(LOG_DEST)

    print("\n" + "=" * 80)
    print("WARMUP PHASE")
    print("=" * 80)

    # Warmup undecorated version
    print("\nWarming up undecorated version...")
    warmup(test_matmul_undecorated, A, B, num_warmup=10)
    print("  Complete.")

    # Warmup decorated version
    print("\nWarming up decorated version...")
    warmup(test_matmul_decorated, A, B, num_warmup=10)
    print("  Complete.")

    print("\n" + "=" * 80)
    print("BENCHMARK PHASE")
    print("=" * 80)

    # Store results
    results = BenchmarkResults()

    # Benchmark undecorated version
    undecorated_times = benchmark_function(
        test_matmul_undecorated, "Undecorated (baseline)", A, B, num_iterations
    )
    results.set_undecorated(undecorated_times)

    # Benchmark decorated version
    decorated_times = benchmark_function(
        test_matmul_decorated,
        f"Decorated (logging level {LOGGING_LEVEL})",
        A,
        B,
        num_iterations,
    )
    results.set_decorated(decorated_times)

    # Print summary
    results.print_summary(LOGGING_LEVEL)

    # Check log file size
    if LOGGING_LEVEL > 0 and os.path.exists(LOG_DEST):
        log_size = os.path.getsize(LOG_DEST)
        print("\n" + "=" * 80)
        print("LOG FILE INFO")
        print("=" * 80)
        print(f"Log file: {LOG_DEST}")
        print(f"Log size: {log_size / 1024:.2f} KB ({log_size} bytes)")
        print(f"Iterations logged: {num_iterations}")
        print(f"Bytes per iteration: {log_size / num_iterations:.2f}")

        # Cleanup option
        cleanup_log = os.environ.get("CLEANUP_LOG", "true").lower() == "true"
        if cleanup_log:
            os.remove(LOG_DEST)
            print("\n Log file removed (set CLEANUP_LOG=false to keep it)")
        else:
            print(f"\n Log file preserved at {LOG_DEST}")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("\nTo benchmark other levels, run:")
    for level in [0, 1, 3, 5]:
        if level != LOGGING_LEVEL:
            print(f"  FLASHINFER_LOGLEVEL={level} python {sys.argv[0]}")

    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
    except Exception as e:
        print(f"\n\nError during benchmark: {e}")
        import traceback

        traceback.print_exc()
