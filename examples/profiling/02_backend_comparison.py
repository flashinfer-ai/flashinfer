"""
Example: Comparing FlashInfer backends for different workloads.

This script demonstrates:
- Comparing multiple backends (FlashInfer, cuDNN, CUTLASS)
- Identifying the best backend for each scenario
- Visualizing performance differences

Run with:
    python examples/profiling/02_backend_comparison.py
"""

import torch
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class BackendResult:
    """Container for backend benchmark results."""
    backend_name: str
    median_latency_ms: float
    std_latency_ms: float
    tflops: float
    success: bool = True
    error: str = ""


class BackendComparator:
    """Compare performance across different backends."""
    
    def __init__(self, num_warmup: int = 5, num_iters: int = 50):
        self.num_warmup = num_warmup
        self.num_iters = num_iters
        self.device = torch.device("cuda:0")
    
    def benchmark_backend(
        self,
        backend_name: str,
        attention_func: callable,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor
    ) -> BackendResult:
        """
        Benchmark a single backend.
        
        Args:
            backend_name: Name of the backend
            attention_func: Attention function to benchmark
            q: Query tensor
            k: Key tensor
            v: Value tensor
        
        Returns:
            BackendResult with performance metrics
        """
        try:
            # Warmup
            for _ in range(self.num_warmup):
                _ = attention_func(q, k, v)
            torch.cuda.synchronize()
            
            # Benchmark
            latencies = []
            for _ in range(self.num_iters):
                start = time.perf_counter()
                output = attention_func(q, k, v)
                torch.cuda.synchronize()
                end = time.perf_counter()
                latencies.append((end - start) * 1000)
            
            # Compute statistics
            latencies_sorted = sorted(latencies)
            n = len(latencies)
            median_latency = latencies_sorted[n // 2]
            mean_latency = sum(latencies) / n
            std_latency = (
                sum((x - mean_latency) ** 2 for x in latencies) / n
            ) ** 0.5
            
            # Compute TFLOPS (approximate)
            qo_heads, head_dim = q.shape
            kv_len, kv_heads, _ = k.shape
            flops = 2 * qo_heads * kv_len * head_dim * kv_heads * 2
            tflops = (flops / (median_latency * 1e-3)) / 1e12
            
            return BackendResult(
                backend_name=backend_name,
                median_latency_ms=median_latency,
                std_latency_ms=std_latency,
                tflops=tflops,
                success=True
            )
        
        except Exception as e:
            return BackendResult(
                backend_name=backend_name,
                median_latency_ms=float('inf'),
                std_latency_ms=0.0,
                tflops=0.0,
                success=False,
                error=str(e)
            )
    
    def compare_backends(
        self,
        config: Dict[str, any]
    ) -> List[BackendResult]:
        """
        Compare multiple backends for a given configuration.
        
        Args:
            config: Dictionary with:
                - num_qo_heads, num_kv_heads, head_dim
                - kv_len
                - dtype
        
        Returns:
            List of BackendResult sorted by performance
        """
        # Create test data
        q = torch.randn(
            config['num_qo_heads'], config['head_dim'],
            device=self.device, dtype=config['dtype']
        )
        k = torch.randn(
            config['kv_len'], config['num_kv_heads'], config['head_dim'],
            device=self.device, dtype=config['dtype']
        )
        v = torch.randn(
            config['kv_len'], config['num_kv_heads'], config['head_dim'],
            device=self.device, dtype=config['dtype']
        )
        
        results = []
        
        # Test FlashInfer
        try:
            import flashinfer
            results.append(
                self.benchmark_backend(
                    "flashinfer",
                    flashinfer.single_decode_with_kv_cache,
                    q, k, v
                )
            )
        except ImportError:
            print("⚠️  FlashInfer not available")
        
        # Test cuDNN (if available)
        try:
            from flashinfer.cudnn import cudnn_decode_with_kv_cache
            results.append(
                self.benchmark_backend(
                    "cudnn",
                    cudnn_decode_with_kv_cache,
                    q, k, v
                )
            )
        except (ImportError, AttributeError):
            print("⚠️  cuDNN backend not available")
        
        # Sort by latency (best first)
        results.sort(key=lambda x: x.median_latency_ms)
        
        return results
    
    def print_comparison(
        self,
        results: List[BackendResult],
        config: Dict[str, any]
    ):
        """Print formatted comparison results."""
        print(f"\n{'='*70}")
        print(f"Backend Comparison - "
              f"KV len: {config['kv_len']}, "
              f"Heads: {config['num_qo_heads']}/{config['num_kv_heads']}, "
              f"Dim: {config['head_dim']}")
        print(f"{'='*70}")
        
        if not results or not any(r.success for r in results):
            print("❌ No successful benchmarks")
            return
        
        # Find best result
        best_result = next(r for r in results if r.success)
        
        print(f"\n{'Backend':<15} {'Latency (ms)':<15} {'TFLOPS':<12} {'vs Best':<10}")
        print(f"{'-'*15} {'-'*15} {'-'*12} {'-'*10}")
        
        for result in results:
            if not result.success:
                print(f"{result.backend_name:<15} {'FAILED':<15} {'-':<12} {'-':<10}")
                continue
            
            speedup_vs_best = best_result.median_latency_ms / result.median_latency_ms
            vs_best_str = f"{speedup_vs_best:.2f}x"
            
            emoji = "🥇" if result == best_result else "  "
            
            print(f"{emoji} {result.backend_name:<13} "
                  f"{result.median_latency_ms:<15.3f} "
                  f"{result.tflops:<12.2f} "
                  f"{vs_best_str:<10}")
        
        # Recommendation
        print(f"\n💡 Recommendation: Use '{best_result.backend_name}' "
              f"for this configuration")


def main():
    """Run backend comparison examples."""
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This example requires a GPU.")
        return
    
    # Print GPU info
    device_name = torch.cuda.get_device_name(0)
    device_props = torch.cuda.get_device_properties(0)
    compute_cap = (device_props.major, device_props.minor)
    
    print(f"🖥️  GPU: {device_name}")
    print(f"🔢  Compute Capability: SM{compute_cap[0]}{compute_cap[1]}")
    
    comparator = BackendComparator(num_warmup=5, num_iters=50)
    
    # Scenario 1: Short context (chatbot)
    print("\n" + "="*70)
    print("Scenario 1: Short Context (Chatbot)")
    print("="*70)
    results_1 = comparator.compare_backends({
        'kv_len': 512,
        'num_qo_heads': 32,
        'num_kv_heads': 8,
        'head_dim': 128,
        'dtype': torch.float16
    })
    comparator.print_comparison(results_1, {
        'kv_len': 512,
        'num_qo_heads': 32,
        'num_kv_heads': 8,
        'head_dim': 128
    })
    
    # Scenario 2: Medium context (document QA)
    print("\n" + "="*70)
    print("Scenario 2: Medium Context (Document QA)")
    print("="*70)
    results_2 = comparator.compare_backends({
        'kv_len': 2048,
        'num_qo_heads': 32,
        'num_kv_heads': 8,
        'head_dim': 128,
        'dtype': torch.float16
    })
    comparator.print_comparison(results_2, {
        'kv_len': 2048,
        'num_qo_heads': 32,
        'num_kv_heads': 8,
        'head_dim': 128
    })
    
    # Scenario 3: Long context (code generation)
    print("\n" + "="*70)
    print("Scenario 3: Long Context (Code Generation)")
    print("="*70)
    results_3 = comparator.compare_backends({
        'kv_len': 8192,
        'num_qo_heads': 32,
        'num_kv_heads': 8,
        'head_dim': 128,
        'dtype': torch.float16
    })
    comparator.print_comparison(results_3, {
        'kv_len': 8192,
        'num_qo_heads': 32,
        'num_kv_heads': 8,
        'head_dim': 128
    })
    
    # Summary
    print("\n" + "="*70)
    print("📊 Summary")
    print("="*70)
    print("\nBackend recommendations by scenario:")
    
    for i, (results, scenario) in enumerate([
        (results_1, "Short context (512)"),
        (results_2, "Medium context (2048)"),
        (results_3, "Long context (8192)")
    ], 1):
        best = next((r for r in results if r.success), None)
        if best:
            print(f"  {i}. {scenario:<25} → {best.backend_name}")
    
    print("\n✅ Comparison complete!")
    print("\nKey insights:")
    print("  • Different backends excel at different workload shapes")
    print("  • cuDNN often better for large batches (not shown here)")
    print("  • FlashInfer provides good all-around performance")
    print("  • Always benchmark your specific use case!")


if __name__ == "__main__":
    main()
