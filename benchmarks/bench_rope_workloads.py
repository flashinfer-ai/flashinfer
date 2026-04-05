"""
Benchmark RoPE CuTe-DSL backend against CUDA for typical LLM inference workloads.

Tests MLA, GQA, and MHA configurations across realistic batch/seq_len combinations.
Tests all 10 RoPE APIs with CuTe-DSL backend support.

Supports dimension sweeps to test:
- dtype: fp16 vs bf16
- interleave: NeoX (False) vs GPT-J (True) style
- rotary_dim: full vs partial (half of head_dim)
- offsets: zero vs non-zero (simulating KV cache append)

Usage:
    # Core workloads (default)
    python bench_rope_workloads.py

    # With dimension sweeps
    python bench_rope_workloads.py --dtype both
    python bench_rope_workloads.py --interleave both
    python bench_rope_workloads.py --all-dimensions

    # Focused benchmarks
    python bench_rope_workloads.py --focus decode
    python bench_rope_workloads.py --api apply_rope_pos_ids --config GQA
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

# Ensure we import from the workspace, not installed package
workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root))

import flashinfer
from flashinfer.testing.utils import bench_gpu_time


# =============================================================================
# Constants
# =============================================================================

# All 10 RoPE APIs
ALL_APIS = [
    "apply_rope",
    "apply_rope_inplace",
    "apply_rope_pos_ids",
    "apply_rope_pos_ids_inplace",
    "apply_llama31_rope",
    "apply_llama31_rope_inplace",
    "apply_llama31_rope_pos_ids",
    "apply_llama31_rope_pos_ids_inplace",
    "apply_rope_with_cos_sin_cache",
    "apply_rope_with_cos_sin_cache_inplace",
]

# API groups for focused benchmarks
API_GROUPS = {
    "standard": [
        "apply_rope",
        "apply_rope_inplace",
        "apply_rope_pos_ids",
        "apply_rope_pos_ids_inplace",
    ],
    "llama31": [
        "apply_llama31_rope",
        "apply_llama31_rope_inplace",
        "apply_llama31_rope_pos_ids",
        "apply_llama31_rope_pos_ids_inplace",
    ],
    "cos_sin_cache": [
        "apply_rope_with_cos_sin_cache",
        "apply_rope_with_cos_sin_cache_inplace",
    ],
}

# Configuration presets
CONFIGS = {
    "MLA": {
        "num_qo_heads": 128,
        "num_kv_heads": 1,
        "head_dim": 64,
    },
    "GQA": {
        "num_qo_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
    },
    "MHA": {
        "num_qo_heads": 32,
        "num_kv_heads": 32,
        "head_dim": 128,
    },
}

# Workload categories
DECODE_WORKLOADS = [
    (1, 1, "Single decode"),
    (32, 1, "Decode batch=32"),
    (128, 1, "Decode batch=128"),
    (256, 1, "Decode batch=256"),
    (512, 1, "Decode batch=512"),
    (1024, 1, "Decode batch=1024"),
    (2048, 1, "Decode batch=2048"),
]

PREFILL_WORKLOADS = [
    (1, 1024, "Prefill 1K"),
    (1, 4096, "Prefill 4K"),
    (1, 8192, "Prefill 8K"),
    (1, 16384, "Prefill 16K"),
    (1, 32768, "Prefill 32K"),
]

BATCHED_PREFILL_WORKLOADS = [
    (4, 2048, "Batch prefill 4x2K"),
    (8, 1024, "Batch prefill 8x1K"),
    (16, 512, "Batch prefill 16x512"),
    (32, 256, "Batch prefill 32x256"),
]

ALL_WORKLOADS = DECODE_WORKLOADS + PREFILL_WORKLOADS + BATCHED_PREFILL_WORKLOADS


# =============================================================================
# Helper Functions
# =============================================================================


def create_test_data(
    batch_size: int,
    seq_len: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    use_nonzero_offsets: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create test tensors for RoPE benchmarking."""
    nnz = batch_size * seq_len

    q = torch.randn(nnz, num_qo_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(nnz, num_kv_heads, head_dim, dtype=dtype, device=device)

    indptr = torch.tensor(
        [i * seq_len for i in range(batch_size + 1)], dtype=torch.int32, device=device
    )

    if use_nonzero_offsets:
        # Simulate KV cache append: random offsets between 0 and 1000
        offsets = torch.randint(
            0, 1000, (batch_size,), dtype=torch.int32, device=device
        )
    else:
        offsets = torch.zeros(batch_size, dtype=torch.int32, device=device)

    pos_ids = torch.arange(seq_len, dtype=torch.int32, device=device).repeat(batch_size)
    if use_nonzero_offsets:
        # Adjust pos_ids based on offsets
        for i in range(batch_size):
            start = i * seq_len
            end = (i + 1) * seq_len
            pos_ids[start:end] += offsets[i]

    return q, k, indptr, offsets, pos_ids


def create_cos_sin_cache(
    max_seq_len: int,
    rotary_dim: int,
    device: str,
    base: float = 10000.0,
) -> torch.Tensor:
    """Create precomputed cos/sin cache for RoPE."""
    freqs = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim // 2, device=device, dtype=torch.float32)
            / (rotary_dim // 2)
        )
    )
    positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    angles = positions.unsqueeze(1) * freqs.unsqueeze(0)
    cos_cache = torch.cos(angles)
    sin_cache = torch.sin(angles)
    cos_sin_cache = torch.cat([cos_cache, sin_cache], dim=1)
    return cos_sin_cache


def benchmark_api(
    api_name: str,
    batch_size: int,
    seq_len: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    interleave: bool = False,
    rotary_dim: Optional[int] = None,
    use_nonzero_offsets: bool = False,
) -> Tuple[float, float]:
    """Benchmark a specific API with both backends."""
    if rotary_dim is None:
        rotary_dim = head_dim

    q, k, indptr, offsets, pos_ids = create_test_data(
        batch_size,
        seq_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        dtype,
        device,
        use_nonzero_offsets=use_nonzero_offsets,
    )

    # Define API-specific benchmark functions
    if api_name == "apply_rope":

        def run_cuda():
            return flashinfer.apply_rope(
                q,
                k,
                indptr,
                offsets,
                rotary_dim=rotary_dim,
                interleave=interleave,
                backend="cuda",
            )

        def run_cute():
            return flashinfer.apply_rope(
                q,
                k,
                indptr,
                offsets,
                rotary_dim=rotary_dim,
                interleave=interleave,
                backend="cute-dsl",
            )

    elif api_name == "apply_rope_inplace":

        def run_cuda():
            q_c, k_c = q.clone(), k.clone()
            flashinfer.apply_rope_inplace(
                q_c,
                k_c,
                indptr,
                offsets,
                rotary_dim=rotary_dim,
                interleave=interleave,
                backend="cuda",
            )

        def run_cute():
            q_c, k_c = q.clone(), k.clone()
            flashinfer.apply_rope_inplace(
                q_c,
                k_c,
                indptr,
                offsets,
                rotary_dim=rotary_dim,
                interleave=interleave,
                backend="cute-dsl",
            )

    elif api_name == "apply_rope_pos_ids":

        def run_cuda():
            return flashinfer.apply_rope_pos_ids(
                q,
                k,
                pos_ids,
                rotary_dim=rotary_dim,
                interleave=interleave,
                backend="cuda",
            )

        def run_cute():
            return flashinfer.apply_rope_pos_ids(
                q,
                k,
                pos_ids,
                rotary_dim=rotary_dim,
                interleave=interleave,
                backend="cute-dsl",
            )

    elif api_name == "apply_rope_pos_ids_inplace":

        def run_cuda():
            q_c, k_c = q.clone(), k.clone()
            flashinfer.apply_rope_pos_ids_inplace(
                q_c,
                k_c,
                pos_ids,
                rotary_dim=rotary_dim,
                interleave=interleave,
                backend="cuda",
            )

        def run_cute():
            q_c, k_c = q.clone(), k.clone()
            flashinfer.apply_rope_pos_ids_inplace(
                q_c,
                k_c,
                pos_ids,
                rotary_dim=rotary_dim,
                interleave=interleave,
                backend="cute-dsl",
            )

    elif api_name == "apply_llama31_rope":

        def run_cuda():
            return flashinfer.apply_llama31_rope(
                q,
                k,
                indptr,
                offsets,
                rotary_dim=rotary_dim,
                interleave=interleave,
                backend="cuda",
            )

        def run_cute():
            return flashinfer.apply_llama31_rope(
                q,
                k,
                indptr,
                offsets,
                rotary_dim=rotary_dim,
                interleave=interleave,
                backend="cute-dsl",
            )

    elif api_name == "apply_llama31_rope_inplace":

        def run_cuda():
            q_c, k_c = q.clone(), k.clone()
            flashinfer.apply_llama31_rope_inplace(
                q_c,
                k_c,
                indptr,
                offsets,
                rotary_dim=rotary_dim,
                interleave=interleave,
                backend="cuda",
            )

        def run_cute():
            q_c, k_c = q.clone(), k.clone()
            flashinfer.apply_llama31_rope_inplace(
                q_c,
                k_c,
                indptr,
                offsets,
                rotary_dim=rotary_dim,
                interleave=interleave,
                backend="cute-dsl",
            )

    elif api_name == "apply_llama31_rope_pos_ids":

        def run_cuda():
            return flashinfer.apply_llama31_rope_pos_ids(
                q,
                k,
                pos_ids,
                rotary_dim=rotary_dim,
                interleave=interleave,
                backend="cuda",
            )

        def run_cute():
            return flashinfer.apply_llama31_rope_pos_ids(
                q,
                k,
                pos_ids,
                rotary_dim=rotary_dim,
                interleave=interleave,
                backend="cute-dsl",
            )

    elif api_name == "apply_llama31_rope_pos_ids_inplace":

        def run_cuda():
            q_c, k_c = q.clone(), k.clone()
            flashinfer.apply_llama31_rope_pos_ids_inplace(
                q_c,
                k_c,
                pos_ids,
                rotary_dim=rotary_dim,
                interleave=interleave,
                backend="cuda",
            )

        def run_cute():
            q_c, k_c = q.clone(), k.clone()
            flashinfer.apply_llama31_rope_pos_ids_inplace(
                q_c,
                k_c,
                pos_ids,
                rotary_dim=rotary_dim,
                interleave=interleave,
                backend="cute-dsl",
            )

    elif api_name == "apply_rope_with_cos_sin_cache":
        nnz = batch_size * seq_len
        query = torch.randn(nnz, num_qo_heads * head_dim, dtype=dtype, device=device)
        key = torch.randn(nnz, num_kv_heads * head_dim, dtype=dtype, device=device)
        positions = torch.arange(seq_len, dtype=torch.int64, device=device).repeat(
            batch_size
        )
        if use_nonzero_offsets:
            for i in range(batch_size):
                start = i * seq_len
                end = (i + 1) * seq_len
                positions[start:end] += torch.randint(
                    0, 1000, (1,), device=device
                ).item()
        max_seq_len_cache = max(seq_len + 1000, 2048)  # Account for potential offsets
        cos_sin_cache = create_cos_sin_cache(max_seq_len_cache, rotary_dim, device)

        def run_cuda():
            return flashinfer.apply_rope_with_cos_sin_cache(
                positions=positions,
                query=query,
                key=key,
                head_size=head_dim,
                cos_sin_cache=cos_sin_cache,
                is_neox=(not interleave),
                backend="cuda",
            )

        def run_cute():
            return flashinfer.apply_rope_with_cos_sin_cache(
                positions=positions,
                query=query,
                key=key,
                head_size=head_dim,
                cos_sin_cache=cos_sin_cache,
                is_neox=(not interleave),
                backend="cute-dsl",
            )

    elif api_name == "apply_rope_with_cos_sin_cache_inplace":
        nnz = batch_size * seq_len
        query = torch.randn(nnz, num_qo_heads * head_dim, dtype=dtype, device=device)
        key = torch.randn(nnz, num_kv_heads * head_dim, dtype=dtype, device=device)
        positions = torch.arange(seq_len, dtype=torch.int64, device=device).repeat(
            batch_size
        )
        if use_nonzero_offsets:
            for i in range(batch_size):
                start = i * seq_len
                end = (i + 1) * seq_len
                positions[start:end] += torch.randint(
                    0, 1000, (1,), device=device
                ).item()
        max_seq_len_cache = max(seq_len + 1000, 2048)
        cos_sin_cache = create_cos_sin_cache(max_seq_len_cache, rotary_dim, device)

        def run_cuda():
            q_c, k_c = query.clone(), key.clone()
            flashinfer.apply_rope_with_cos_sin_cache_inplace(
                positions=positions,
                query=q_c,
                key=k_c,
                head_size=head_dim,
                cos_sin_cache=cos_sin_cache,
                is_neox=(not interleave),
                backend="cuda",
            )

        def run_cute():
            q_c, k_c = query.clone(), key.clone()
            flashinfer.apply_rope_with_cos_sin_cache_inplace(
                positions=positions,
                query=q_c,
                key=k_c,
                head_size=head_dim,
                cos_sin_cache=cos_sin_cache,
                is_neox=(not interleave),
                backend="cute-dsl",
            )

    else:
        raise ValueError(f"Unknown API: {api_name}")

    cuda_times = bench_gpu_time(run_cuda)
    cuda_ms = np.median(cuda_times)

    cute_times = bench_gpu_time(run_cute)
    cute_ms = np.median(cute_times)

    return cuda_ms, cute_ms


# =============================================================================
# Benchmark Runners
# =============================================================================


def run_workload_benchmark(
    api_name: str,
    config: dict,
    workloads: List[Tuple[int, int, str]],
    dtype: torch.dtype = torch.float16,
    interleave: bool = False,
    rotary_dim: Optional[int] = None,
    use_nonzero_offsets: bool = False,
) -> list:
    """Run benchmark for a specific API and configuration."""
    print(
        f"{'Workload':<25} | {'Batch':>6} | {'Seq':>6} | {'Tokens':>10} | "
        f"{'CUDA (ms)':>10} | {'CuTe-DSL':>10} | {'Speedup':>8}"
    )
    print("-" * 95)

    results = []
    for batch_size, seq_len, desc in workloads:
        nnz = batch_size * seq_len

        try:
            cuda_ms, cute_ms = benchmark_api(
                api_name=api_name,
                batch_size=batch_size,
                seq_len=seq_len,
                num_qo_heads=config["num_qo_heads"],
                num_kv_heads=config["num_kv_heads"],
                head_dim=config["head_dim"],
                dtype=dtype,
                interleave=interleave,
                rotary_dim=rotary_dim,
                use_nonzero_offsets=use_nonzero_offsets,
            )
            speedup = cuda_ms / cute_ms
            status = ""
            if speedup < 0.95:
                status = " ⚠️"
            elif speedup > 1.1:
                status = " 🚀"

            print(
                f"{desc:<25} | {batch_size:>6} | {seq_len:>6} | {nnz:>10,} | "
                f"{cuda_ms:>10.4f} | {cute_ms:>10.4f} | {speedup:>7.2f}x{status}"
            )
            results.append((desc, batch_size, seq_len, nnz, cuda_ms, cute_ms, speedup))
        except Exception as e:
            print(
                f"{desc:<25} | {batch_size:>6} | {seq_len:>6} | {nnz:>10,} | "
                f"{'ERROR':>10} | {'ERROR':>10} | {str(e)[:20]}"
            )

    if results:
        speedups = [r[6] for r in results]
        print("-" * 95)
        print(
            f"{'Summary':<25} | {'':>6} | {'':>6} | {'':>10} | "
            f"{'':>10} | {'':>10} | "
            f"min={min(speedups):.2f}x max={max(speedups):.2f}x avg={np.mean(speedups):.2f}x"
        )

    return results


def run_dimension_sweep(
    dimension: str,
    apis: List[str],
    config_name: str,
    config: dict,
    workload: Tuple[int, int, str],
) -> dict:
    """Run a dimension sweep benchmark."""
    batch_size, seq_len, desc = workload
    results = {}

    if dimension == "dtype":
        dtypes = [("fp16", torch.float16), ("bf16", torch.bfloat16)]
        print(
            f"\n{'API':<40} | {'fp16 CUDA':>10} | {'fp16 CuTe':>10} | {'bf16 CUDA':>10} | {'bf16 CuTe':>10}"
        )
        print("-" * 95)

        for api_name in apis:
            row = {"api": api_name}
            try:
                for dtype_name, dtype in dtypes:
                    cuda_ms, cute_ms = benchmark_api(
                        api_name=api_name,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        num_qo_heads=config["num_qo_heads"],
                        num_kv_heads=config["num_kv_heads"],
                        head_dim=config["head_dim"],
                        dtype=dtype,
                    )
                    row[f"{dtype_name}_cuda"] = cuda_ms
                    row[f"{dtype_name}_cute"] = cute_ms

                print(
                    f"{api_name:<40} | {row['fp16_cuda']:>10.4f} | {row['fp16_cute']:>10.4f} | "
                    f"{row['bf16_cuda']:>10.4f} | {row['bf16_cute']:>10.4f}"
                )
                results[api_name] = row
            except Exception as e:
                print(f"{api_name:<40} | ERROR: {str(e)[:50]}")

    elif dimension == "interleave":
        print(
            f"\n{'API':<40} | {'NeoX CUDA':>10} | {'NeoX CuTe':>10} | {'GPT-J CUDA':>10} | {'GPT-J CuTe':>10}"
        )
        print("-" * 95)

        for api_name in apis:
            # Skip cos_sin_cache APIs for interleave sweep (they use is_neox parameter differently)
            if "cos_sin_cache" in api_name:
                continue
            row = {"api": api_name}
            try:
                for mode_name, interleave in [("neox", False), ("gptj", True)]:
                    cuda_ms, cute_ms = benchmark_api(
                        api_name=api_name,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        num_qo_heads=config["num_qo_heads"],
                        num_kv_heads=config["num_kv_heads"],
                        head_dim=config["head_dim"],
                        interleave=interleave,
                    )
                    row[f"{mode_name}_cuda"] = cuda_ms
                    row[f"{mode_name}_cute"] = cute_ms

                print(
                    f"{api_name:<40} | {row['neox_cuda']:>10.4f} | {row['neox_cute']:>10.4f} | "
                    f"{row['gptj_cuda']:>10.4f} | {row['gptj_cute']:>10.4f}"
                )
                results[api_name] = row
            except Exception as e:
                print(f"{api_name:<40} | ERROR: {str(e)[:50]}")

    elif dimension == "rotary_dim":
        head_dim = config["head_dim"]
        dims = [("full", head_dim), ("half", head_dim // 2)]
        print(
            f"\n{'API':<40} | {'Full CUDA':>10} | {'Full CuTe':>10} | {'Half CUDA':>10} | {'Half CuTe':>10}"
        )
        print("-" * 95)

        for api_name in apis:
            row = {"api": api_name}
            try:
                for dim_name, rotary_dim in dims:
                    cuda_ms, cute_ms = benchmark_api(
                        api_name=api_name,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        num_qo_heads=config["num_qo_heads"],
                        num_kv_heads=config["num_kv_heads"],
                        head_dim=head_dim,
                        rotary_dim=rotary_dim,
                    )
                    row[f"{dim_name}_cuda"] = cuda_ms
                    row[f"{dim_name}_cute"] = cute_ms

                print(
                    f"{api_name:<40} | {row['full_cuda']:>10.4f} | {row['full_cute']:>10.4f} | "
                    f"{row['half_cuda']:>10.4f} | {row['half_cute']:>10.4f}"
                )
                results[api_name] = row
            except Exception as e:
                print(f"{api_name:<40} | ERROR: {str(e)[:50]}")

    elif dimension == "offsets":
        print(
            f"\n{'API':<40} | {'Zero CUDA':>10} | {'Zero CuTe':>10} | {'NonZ CUDA':>10} | {'NonZ CuTe':>10}"
        )
        print("-" * 95)

        for api_name in apis:
            row = {"api": api_name}
            try:
                for offset_name, use_offsets in [("zero", False), ("nonzero", True)]:
                    cuda_ms, cute_ms = benchmark_api(
                        api_name=api_name,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        num_qo_heads=config["num_qo_heads"],
                        num_kv_heads=config["num_kv_heads"],
                        head_dim=config["head_dim"],
                        use_nonzero_offsets=use_offsets,
                    )
                    row[f"{offset_name}_cuda"] = cuda_ms
                    row[f"{offset_name}_cute"] = cute_ms

                print(
                    f"{api_name:<40} | {row['zero_cuda']:>10.4f} | {row['zero_cute']:>10.4f} | "
                    f"{row['nonzero_cuda']:>10.4f} | {row['nonzero_cute']:>10.4f}"
                )
                results[api_name] = row
            except Exception as e:
                print(f"{api_name:<40} | ERROR: {str(e)[:50]}")

    return results


def run_full_benchmark(
    apis: List[str],
    configs: dict,
    workloads: List[Tuple[int, int, str]],
    dtype: torch.dtype = torch.float16,
    interleave: bool = False,
    rotary_dim: Optional[int] = None,
    use_nonzero_offsets: bool = False,
):
    """Run full benchmark for specified APIs and configurations."""
    dtype_str = "bf16" if dtype == torch.bfloat16 else "fp16"
    interleave_str = "GPT-J (interleaved)" if interleave else "NeoX (non-interleaved)"
    rotary_str = f"rotary_dim={rotary_dim}" if rotary_dim else "rotary_dim=full"
    offset_str = "non-zero offsets" if use_nonzero_offsets else "zero offsets"

    all_api_results = {}

    for api_name in apis:
        print(f"\n{'#' * 95}")
        print(f"# API: {api_name}")
        print(f"# Settings: {dtype_str}, {interleave_str}, {rotary_str}, {offset_str}")
        print(f"{'#' * 95}")

        api_results = {}
        for config_name, config in configs.items():
            print(f"\n{'=' * 95}")
            print(f"Configuration: {config_name}")
            print(
                f"  num_qo_heads={config['num_qo_heads']}, num_kv_heads={config['num_kv_heads']}, head_dim={config['head_dim']}"
            )
            print(f"{'=' * 95}")

            effective_rotary_dim = rotary_dim if rotary_dim else config["head_dim"]
            api_results[config_name] = run_workload_benchmark(
                api_name,
                config,
                workloads,
                dtype=dtype,
                interleave=interleave,
                rotary_dim=effective_rotary_dim,
                use_nonzero_offsets=use_nonzero_offsets,
            )

        # Summary for this API
        print(f"\n{'-' * 95}")
        print(f"Summary for {api_name}:")
        for config_name, results in api_results.items():
            if results:
                speedups = [r[6] for r in results]
                print(
                    f"  {config_name}: min={min(speedups):.2f}x, max={max(speedups):.2f}x, avg={np.mean(speedups):.2f}x"
                )

        all_api_results[api_name] = api_results

    # Overall summary
    print("\n" + "=" * 95)
    print("OVERALL SUMMARY")
    print("=" * 95)
    for api_name, config_results in all_api_results.items():
        all_speedups = []
        for _, results in config_results.items():
            if results:
                all_speedups.extend([r[6] for r in results])
        if all_speedups:
            print(
                f"{api_name}: min={min(all_speedups):.2f}x, max={max(all_speedups):.2f}x, avg={np.mean(all_speedups):.2f}x"
            )

    return all_api_results


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark RoPE APIs with dimension sweeps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Core workloads (default)
  python bench_rope_workloads.py

  # Specific API and config
  python bench_rope_workloads.py --api apply_rope_pos_ids --config GQA

  # Dimension sweeps
  python bench_rope_workloads.py --dtype both
  python bench_rope_workloads.py --interleave both
  python bench_rope_workloads.py --rotary-dim both
  python bench_rope_workloads.py --offsets both
  python bench_rope_workloads.py --all-dimensions

  # Focused workloads
  python bench_rope_workloads.py --focus decode
  python bench_rope_workloads.py --focus prefill
  python bench_rope_workloads.py --focus llama31
        """,
    )

    # API selection
    parser.add_argument(
        "--api",
        type=str,
        default="all",
        choices=ALL_APIS + ["all", "standard", "llama31", "cos_sin_cache"],
        help="Which API(s) to benchmark",
    )

    # Config selection
    parser.add_argument(
        "--config",
        type=str,
        default="all",
        choices=["MLA", "GQA", "MHA", "all"],
        help="Which configuration to benchmark",
    )

    # Focus selection
    parser.add_argument(
        "--focus",
        type=str,
        default=None,
        choices=["decode", "prefill", "batched"],
        help="Focus on specific workload category",
    )

    # Dimension sweeps
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "both"],
        help="Data type(s) to benchmark",
    )
    parser.add_argument(
        "--interleave",
        type=str,
        default="false",
        choices=["true", "false", "both"],
        help="Interleave mode(s) to benchmark (NeoX=false, GPT-J=true)",
    )
    parser.add_argument(
        "--rotary-dim",
        type=str,
        default="full",
        choices=["full", "half", "both"],
        help="Rotary dimension(s) to benchmark",
    )
    parser.add_argument(
        "--offsets",
        type=str,
        default="zero",
        choices=["zero", "nonzero", "both"],
        help="Offset mode(s) to benchmark",
    )
    parser.add_argument(
        "--all-dimensions",
        action="store_true",
        help="Run all dimension sweeps",
    )

    args = parser.parse_args()

    print("=" * 95)
    print("RoPE Backend Benchmark: Typical LLM Inference Workloads")
    print("=" * 95)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA device: {torch.cuda.get_device_name()}")

    # Determine APIs
    if args.api == "all":
        apis = ALL_APIS
    elif args.api in API_GROUPS:
        apis = API_GROUPS[args.api]
    else:
        apis = [args.api]

    # Determine configs
    if args.config == "all":
        configs = CONFIGS
    else:
        configs = {args.config: CONFIGS[args.config]}

    # Determine workloads
    if args.focus == "decode":
        workloads = DECODE_WORKLOADS
    elif args.focus == "prefill":
        workloads = PREFILL_WORKLOADS
    elif args.focus == "batched":
        workloads = BATCHED_PREFILL_WORKLOADS
    else:
        workloads = ALL_WORKLOADS

    # Check if any dimension sweep is requested
    run_sweeps = args.all_dimensions or any(
        [
            args.dtype == "both",
            args.interleave == "both",
            args.rotary_dim == "both",
            args.offsets == "both",
        ]
    )

    # Run core benchmark with specified settings
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    interleave = args.interleave == "true"
    rotary_dim = None  # Will use head_dim if not specified
    if args.rotary_dim == "half":
        # Will be set per-config in the benchmark
        pass
    use_nonzero_offsets = args.offsets == "nonzero"

    # Run core benchmark
    if not run_sweeps or args.dtype != "both":
        run_full_benchmark(
            apis=apis,
            configs=configs,
            workloads=workloads,
            dtype=dtype,
            interleave=interleave,
            rotary_dim=rotary_dim,
            use_nonzero_offsets=use_nonzero_offsets,
        )

    # Run dimension sweeps if requested
    if run_sweeps:
        sweep_config_name = "GQA"  # Use GQA as representative config for sweeps
        sweep_config = CONFIGS[sweep_config_name]
        sweep_workload = (256, 1, "Decode batch=256")  # Representative workload

        dimensions_to_sweep = []
        if args.all_dimensions or args.dtype == "both":
            dimensions_to_sweep.append("dtype")
        if args.all_dimensions or args.interleave == "both":
            dimensions_to_sweep.append("interleave")
        if args.all_dimensions or args.rotary_dim == "both":
            dimensions_to_sweep.append("rotary_dim")
        if args.all_dimensions or args.offsets == "both":
            dimensions_to_sweep.append("offsets")

        for dimension in dimensions_to_sweep:
            print(f"\n{'=' * 95}")
            print(f"DIMENSION SWEEP: {dimension.upper()}")
            print(f"Config: {sweep_config_name}, Workload: {sweep_workload[2]}")
            print(f"{'=' * 95}")

            run_dimension_sweep(
                dimension, apis, sweep_config_name, sweep_config, sweep_workload
            )


if __name__ == "__main__":
    main()
