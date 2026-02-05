"""
Benchmark: Gated Delta Rule CuTe-DSL Kernel

Simple benchmark showing duration across batch sizes and sequence lengths (T=1,2,3,4).
"""

import math
import statistics
import torch


def get_l2_cache_size():
    """Get L2 cache size in bytes for the current GPU."""
    return torch.cuda.get_device_properties(0).L2_cache_size


def benchmark(
    func, num_iterations=100, n_warmup=10, flush_l2=True, use_dummy_matmul=True
):
    """
    Benchmark a kernel with L2 flushing and return median time in microseconds.

    Args:
        func: Function to benchmark
        num_iterations: Number of timed iterations
        n_warmup: Number of warmup iterations
        flush_l2: Whether to flush L2 cache before each iteration
        use_dummy_matmul: Whether to use dummy matmul for short-lived kernels
    """
    l2_size = get_l2_cache_size()
    cache_flush = torch.empty(l2_size, dtype=torch.uint8, device="cuda")

    # Dummy matmul for short-lived kernels (fills GPU pipeline so CUDA events record properly)
    if use_dummy_matmul:
        A = torch.randn(4096, 4096, dtype=torch.float32, device="cuda")
        B = torch.randn(4096, 4096, dtype=torch.float32, device="cuda")
        _ = A @ B  # Warm up cuBLAS

    # Warmup
    for _ in range(n_warmup):
        if flush_l2:
            cache_flush.zero_()
        func()
    torch.cuda.synchronize()

    # Benchmark
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)]

    for i in range(num_iterations):
        if flush_l2:
            cache_flush.zero_()
        if use_dummy_matmul:
            _ = A @ B  # Dummy work to ensure events record properly for short kernels
        start_events[i].record()
        func()
        end_events[i].record()

    torch.cuda.synchronize()
    times_us = [
        s.elapsed_time(e) * 1000 for s, e in zip(start_events, end_events, strict=True)
    ]
    return statistics.median(times_us)


def create_inputs(B, T, H=16, HV=32, K=128, V=128):
    """Create test inputs."""
    return {
        "q": torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16),
        "k": torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16),
        "v": torch.randn(B, T, HV, V, device="cuda", dtype=torch.bfloat16),
        "a": torch.randn(B, T, HV, device="cuda", dtype=torch.bfloat16) * 0.1,
        "b": torch.randn(B, T, HV, device="cuda", dtype=torch.bfloat16),
        "A_log": torch.randn(HV, device="cuda", dtype=torch.float32) * 0.1,
        "dt_bias": torch.randn(HV, device="cuda", dtype=torch.float32) * 0.1,
        "state": torch.randn(B, HV, V, K, device="cuda", dtype=torch.bfloat16),
        "scale": 1.0 / math.sqrt(K),
    }


def main():
    from gated_delta_rule import gated_delta_rule

    print("=" * 70)
    print("Gated Delta Rule CuTe-DSL Kernel Benchmark")
    print("Config: H=16, HV=32, K=128, V=128, bfloat16")
    print("=" * 70)

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    seqlens = [1, 2, 3, 4]
    num_iterations = 100

    # Results storage
    results = {T: {} for T in seqlens}

    # Benchmark each configuration
    for T in seqlens:
        print(f"\nCompiling and benchmarking T={T}...")
        for B in batch_sizes:
            inputs = create_inputs(B, T)
            state = inputs["state"].clone()

            # Warmup / compile
            _ = gated_delta_rule(
                A_log=inputs["A_log"],
                a=inputs["a"],
                dt_bias=inputs["dt_bias"],
                q=inputs["q"],
                k=inputs["k"],
                v=inputs["v"],
                b=inputs["b"],
                initial_state_source=state,
                scale=inputs["scale"],
            )

            def run_kernel():
                return gated_delta_rule(
                    A_log=inputs["A_log"],
                    a=inputs["a"],
                    dt_bias=inputs["dt_bias"],
                    q=inputs["q"],
                    k=inputs["k"],
                    v=inputs["v"],
                    b=inputs["b"],
                    initial_state_source=state,
                    scale=inputs["scale"],
                )

            time_us = benchmark(
                run_kernel,
                num_iterations=num_iterations,
                flush_l2=True,
                use_dummy_matmul=True,
            )
            results[T][B] = time_us
            print(f"  B={B:>3}: {time_us:>7.1f} us")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Duration (us) by Batch Size and Sequence Length")
    print("=" * 70)

    # Header
    header = f"{'B':>6} |"
    for T in seqlens:
        header += f"   T={T}   |"
    print(header)
    print("-" * 70)

    # Data rows
    for B in batch_sizes:
        row = f"{B:>6} |"
        for T in seqlens:
            row += f" {results[T][B]:>7.1f} |"
        print(row)

    print("-" * 70)

    # Averages
    print("\nAverage duration per T:")
    for T in seqlens:
        avg = sum(results[T].values()) / len(results[T])
        print(f"  T={T}: {avg:.1f} us")


if __name__ == "__main__":
    main()
