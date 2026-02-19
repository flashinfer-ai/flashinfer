"""
Example: Basic performance profiling for FlashInfer decode attention.

This script demonstrates how to:
- Profile attention kernels
- Collect performance metrics
- Analyze results
- Detect bottlenecks

Run with:
    python examples/profiling/01_decode_profiling.py
"""

import torch
import flashinfer
import time
from typing import Dict, List
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class AttentionProfiler:
    """Simple profiler for attention operations."""
    
    def __init__(self, num_warmup: int = 10, num_iters: int = 100):
        self.num_warmup = num_warmup
        self.num_iters = num_iters
        self.device = torch.device("cuda:0")
        
    def profile_decode_attention(
        self,
        batch_size: int,
        kv_len: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16
    ) -> Dict[str, float]:
        """
        Profile a decode attention configuration.
        
        Args:
            batch_size: Number of requests in batch
            kv_len: Length of KV cache
            num_qo_heads: Number of query/output heads
            num_kv_heads: Number of key/value heads (for GQA)
            head_dim: Dimension per head
            dtype: Data type (float16, bfloat16, float8_e4m3fn)
        
        Returns:
            Dictionary with performance metrics
        """
        print(f"\n{'='*70}")
        print(f"Profiling Decode Attention")
        print(f"  Batch size: {batch_size}, KV len: {kv_len}")
        print(f"  QO heads: {num_qo_heads}, KV heads: {num_kv_heads}")
        print(f"  Head dim: {head_dim}, dtype: {dtype}")
        print(f"{'='*70}")
        
        # Create test data
        q = torch.randn(
            num_qo_heads, head_dim,
            device=self.device, dtype=dtype
        )
        k = torch.randn(
            kv_len, num_kv_heads, head_dim,
            device=self.device, dtype=dtype
        )
        v = torch.randn(
            kv_len, num_kv_heads, head_dim,
            device=self.device, dtype=dtype
        )
        
        # Warmup
        print(f"Warming up ({self.num_warmup} iterations)...")
        for _ in range(self.num_warmup):
            _ = flashinfer.single_decode_with_kv_cache(q, k, v)
        torch.cuda.synchronize()
        
        # Benchmark
        print(f"Benchmarking ({self.num_iters} iterations)...")
        latencies = []
        
        for _ in range(self.num_iters):
            start = time.perf_counter()
            output = flashinfer.single_decode_with_kv_cache(q, k, v)
            torch.cuda.synchronize()
            end = time.perf_counter()
            
            latencies.append((end - start) * 1000)  # Convert to ms
        
        # Compute statistics
        latencies_sorted = sorted(latencies)
        n = len(latencies)
        
        median_latency = latencies_sorted[n // 2]
        p95_latency = latencies_sorted[int(0.95 * n)]
        p99_latency = latencies_sorted[int(0.99 * n)]
        mean_latency = sum(latencies) / n
        std_latency = (
            sum((x - mean_latency) ** 2 for x in latencies) / n
        ) ** 0.5
        
        # Compute throughput metrics
        # For decode: one FMA per (qo_head, kv_head, kv_pos, head_dim)
        # Attention: QK^T (qo_heads x kv_heads x kv_len x head_dim operations)
        #           + Softmax
        #           + PV (qo_heads x kv_heads x kv_len x head_dim operations)
        # Approximate as 2 * qo_heads * kv_len * head_dim FMAs per kv_head
        flops_per_iter = (
            2 * num_qo_heads * kv_len * head_dim * num_kv_heads * 2
        )
        tflops = (flops_per_iter / (median_latency * 1e-3)) / 1e12
        
        # Memory bandwidth (rough estimate)
        # Read: Q (qo_heads * head_dim), K (kv_len * kv_heads * head_dim),
        #       V (kv_len * kv_heads * head_dim)
        # Write: O (qo_heads * head_dim)
        bytes_per_element = 2 if dtype == torch.float16 else 4
        memory_bytes = (
            (num_qo_heads * head_dim +  # Q
             kv_len * num_kv_heads * head_dim +  # K
             kv_len * num_kv_heads * head_dim +  # V
             num_qo_heads * head_dim) *  # O
            bytes_per_element
        )
        bandwidth_gb_per_sec = (
            memory_bytes / (median_latency * 1e-3)
        ) / 1e9
        
        results = {
            "median_latency_ms": median_latency,
            "mean_latency_ms": mean_latency,
            "std_latency_ms": std_latency,
            "p95_latency_ms": p95_latency,
            "p99_latency_ms": p99_latency,
            "tflops": tflops,
            "bandwidth_gb_per_sec": bandwidth_gb_per_sec,
            "cv_percent": (std_latency / mean_latency) * 100,  # Coefficient of variation
        }
        
        # Print results
        print(f"\n📊 Results:")
        print(f"  Median latency:    {median_latency:.3f} ms")
        print(f"  Mean ± std:        {mean_latency:.3f} ± {std_latency:.3f} ms")
        print(f"  P95 latency:       {p95_latency:.3f} ms")
        print(f"  P99 latency:       {p99_latency:.3f} ms")
        print(f"  Throughput:        {tflops:.2f} TFLOPS")
        print(f"  Memory bandwidth:  {bandwidth_gb_per_sec:.2f} GB/s")
        print(f"  Consistency (CV):  {results['cv_percent']:.1f}%")
        
        if results['cv_percent'] > 5.0:
            print(f"  ⚠️  High variance detected! Check for thermal throttling.")
        
        return results


def main():
    """Run decode attention profiling examples."""
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This example requires a GPU.")
        return
    
    # Print GPU info
    device_name = torch.cuda.get_device_name(0)
    device_props = torch.cuda.get_device_properties(0)
    compute_cap = (device_props.major, device_props.minor)
    
    print(f"🖥️  GPU: {device_name}")
    print(f"🔢  Compute Capability: SM{compute_cap[0]}{compute_cap[1]}")
    print(f"💾  Memory: {device_props.total_memory / (1024**3):.1f} GB")
    
    profiler = AttentionProfiler(num_warmup=10, num_iters=100)
    
    # Example 1: Small batch, short context
    print("\n" + "="*70)
    print("Example 1: Small Batch, Short Context (Chat scenario)")
    print("="*70)
    results_1 = profiler.profile_decode_attention(
        batch_size=1,
        kv_len=512,
        num_qo_heads=32,
        num_kv_heads=8,  # GQA 4:1
        head_dim=128,
        dtype=torch.float16
    )
    
    # Example 2: Medium batch, medium context
    print("\n" + "="*70)
    print("Example 2: Medium Batch, Medium Context (Typical serving)")
    print("="*70)
    results_2 = profiler.profile_decode_attention(
        batch_size=1,
        kv_len=2048,
        num_qo_heads=32,
        num_kv_heads=8,
        head_dim=128,
        dtype=torch.float16
    )
    
    # Example 3: Large batch, long context
    print("\n" + "="*70)
    print("Example 3: Single Request, Long Context (Long document)")
    print("="*70)
    results_3 = profiler.profile_decode_attention(
        batch_size=1,
        kv_len=8192,
        num_qo_heads=32,
        num_kv_heads=8,
        head_dim=128,
        dtype=torch.float16
    )
    
    # Summary comparison
    print("\n" + "="*70)
    print("📈 Summary Comparison")
    print("="*70)
    print(f"{'Config':<30} {'Latency (ms)':<15} {'TFLOPS':<10}")
    print(f"{'-'*30} {'-'*15} {'-'*10}")
    print(f"{'Small (512 ctx)':<30} "
          f"{results_1['median_latency_ms']:<15.3f} "
          f"{results_1['tflops']:<10.2f}")
    print(f"{'Medium (2048 ctx)':<30} "
          f"{results_2['median_latency_ms']:<15.3f} "
          f"{results_2['tflops']:<10.2f}")
    print(f"{'Long (8192 ctx)':<30} "
          f"{results_3['median_latency_ms']:<15.3f} "
          f"{results_3['tflops']:<10.2f}")
    
    # Scaling analysis
    print("\n📊 Scaling Analysis:")
    ratio_2_1 = results_2['median_latency_ms'] / results_1['median_latency_ms']
    ratio_3_2 = results_3['median_latency_ms'] / results_2['median_latency_ms']
    print(f"  2048/512 latency ratio:  {ratio_2_1:.2f}x (expected: ~4x)")
    print(f"  8192/2048 latency ratio: {ratio_3_2:.2f}x (expected: ~4x)")
    
    if ratio_2_1 < 3.5:
        print("  ✅ Good scaling for medium context")
    else:
        print("  ⚠️  Suboptimal scaling - investigate memory bottlenecks")
    
    print("\n✅ Profiling complete!")
    print("\nNext steps:")
    print("  1. Try different data types (bfloat16, fp8)")
    print("  2. Test with batched attention (BatchDecodeWrapper)")
    print("  3. Profile with CUPTI for detailed metrics")
    print("  4. Compare against other backends (cuDNN, TensorRT-LLM)")


if __name__ == "__main__":
    main()
