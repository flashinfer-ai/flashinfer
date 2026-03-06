"""
Benchmark concat_mla_k kernel for MLA attention.

This benchmark compares different implementations of the concat_mla_k operation:
- torch: Native PyTorch implementation
- torch_compiled: torch.compile optimized version
- flashinfer: FlashInfer CUDA kernel

Supported dtypes: bfloat16, float8_e4m3fn, float8_e5m2

Usage:
$ python bench_concat_mla.py
"""

import numpy as np
import torch

from flashinfer.concat_ops import concat_mla_k as concat_mla_k_flashinfer
from flashinfer.testing.utils import bench_gpu_time

# MLA configuration
NUM_LOCAL_HEADS = 128
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64

BENCHMARK_DTYPES = [
    torch.bfloat16,
    torch.float8_e4m3fn,
]


def _is_fp8(dtype):
    return dtype in (torch.float8_e4m3fn, torch.float8_e5m2)


def _make_random(shape, dtype, device):
    """Create a random tensor, handling FP8 types that don't support randn."""
    if _is_fp8(dtype):
        return torch.randn(shape, dtype=torch.bfloat16, device=device).to(dtype)
    return torch.randn(shape, dtype=dtype, device=device)


def create_data(
    num_tokens: int, dtype: torch.dtype = torch.bfloat16, device: str = "cuda"
):
    """Create test data with potentially non-contiguous tensors."""
    # Create containers with extra space to test strided access
    k_nope_container = _make_random(
        (num_tokens, NUM_LOCAL_HEADS, QK_NOPE_HEAD_DIM + 128),
        dtype=dtype,
        device=device,
    )
    k_nope = k_nope_container[:, :, :QK_NOPE_HEAD_DIM]

    k_rope_container = _make_random(
        (num_tokens, 1, 128 + QK_ROPE_HEAD_DIM),
        dtype=dtype,
        device=device,
    )
    k_rope = k_rope_container[:, :, -QK_ROPE_HEAD_DIM:]

    k = torch.empty(
        (num_tokens, NUM_LOCAL_HEADS, QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM),
        dtype=dtype,
        device=device,
    )
    return dict(k=k, k_nope=k_nope, k_rope=k_rope)


def fn_torch(k: torch.Tensor, k_nope: torch.Tensor, k_rope: torch.Tensor) -> None:
    """Native PyTorch implementation."""
    k[..., :QK_NOPE_HEAD_DIM] = k_nope
    k[..., QK_NOPE_HEAD_DIM:] = k_rope


@torch.compile(dynamic=True)
def fn_torch_compiled(
    k: torch.Tensor, k_nope: torch.Tensor, k_rope: torch.Tensor
) -> None:
    """torch.compile optimized implementation."""
    return fn_torch(k, k_nope, k_rope)


def fn_flashinfer(k: torch.Tensor, k_nope: torch.Tensor, k_rope: torch.Tensor) -> None:
    """FlashInfer CUDA kernel implementation."""
    concat_mla_k_flashinfer(k, k_nope, k_rope)


def execute_and_get_output(f, data):
    """Execute function and return output for correctness checking."""
    data["k"].zero_()
    f(**data)
    # FP8 types don't support .sum(), so check via raw bytes
    assert data["k"].view(torch.uint8).any(), "Output should not be all zeros"
    return data["k"].clone()


def verify_correctness():
    """Verify that all implementations produce the same output."""
    print("Verifying correctness...")
    for dtype in BENCHMARK_DTYPES:
        torch.manual_seed(0)
        data = create_data(num_tokens=32768, dtype=dtype)

        output_ref = execute_and_get_output(fn_torch, data)
        output_flashinfer = execute_and_get_output(fn_flashinfer, data)

        # FP8 types don't support torch.allclose / subtraction directly
        if _is_fp8(dtype):
            matches = output_ref.view(torch.uint8) == output_flashinfer.view(
                torch.uint8
            )
            if not matches.all():
                raise AssertionError(
                    f"[{dtype}] FlashInfer output mismatch! "
                    f"num_mismatches={torch.sum(~matches).item()}"
                )
        else:
            if not torch.allclose(output_ref, output_flashinfer):
                abs_delta = torch.abs(output_ref - output_flashinfer)
                raise AssertionError(
                    f"[{dtype}] FlashInfer output mismatch! "
                    f"abs_delta max={abs_delta.max().item()}, "
                    f"num_mismatches={torch.sum(abs_delta != 0).item()}"
                )
        print(f"  {dtype}: OK")

    print("All implementations produce correct results!")


def benchmark():
    """Run benchmark for all implementations."""
    num_tokens_list = [2048, 4096, 8192, 16384, 32768]

    for dtype in BENCHMARK_DTYPES:
        print(f"\n{'=' * 60}")
        print(f"Benchmarking dtype={dtype}")
        print(f"{'=' * 60}")

        providers = [
            ("torch", fn_torch),
            ("torch_compiled", fn_torch_compiled),
            ("flashinfer", fn_flashinfer),
        ]

        # Warmup torch_compiled
        print("Warming up torch.compile...")
        data = create_data(num_tokens=2048, dtype=dtype)
        for _ in range(3):
            fn_torch_compiled(**data)

        # Collect results
        results = {name: [] for name, _ in providers}

        for num_tokens in num_tokens_list:
            data = create_data(num_tokens=num_tokens, dtype=dtype)
            for name, fn in providers:
                measurements = bench_gpu_time(lambda: fn(**data))
                median_ms = np.median(measurements)
                results[name].append(median_ms)

        # Print header
        header = ["num_tokens"] + [name for name, _ in providers]
        print(" ".join(f"{h:>14}" for h in header))

        # Print rows
        for i, num_tokens in enumerate(num_tokens_list):
            row = [f"{float(num_tokens):>14.1f}"]
            for name, _ in providers:
                row.append(f"{results[name][i]:>14.6f}")
            print(" ".join(row))


if __name__ == "__main__":
    verify_correctness()
    print()
    benchmark()
