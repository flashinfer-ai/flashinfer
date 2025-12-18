"""
Benchmark concat_mla_k kernel for MLA attention.

This benchmark compares different implementations of the concat_mla_k operation:
- torch: Native PyTorch implementation
- torch_compiled: torch.compile optimized version
- flashinfer: FlashInfer CUDA kernel

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


def create_data(num_tokens: int, device: str = "cuda"):
    """Create test data with potentially non-contiguous tensors."""
    # Create containers with extra space to test strided access
    k_nope_container = torch.randn(
        (num_tokens, NUM_LOCAL_HEADS, QK_NOPE_HEAD_DIM + 128),
        dtype=torch.bfloat16,
        device=device,
    )
    k_nope = k_nope_container[:, :, :QK_NOPE_HEAD_DIM]

    k_rope_container = torch.randn(
        (num_tokens, 1, 128 + QK_ROPE_HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    k_rope = k_rope_container[:, :, -QK_ROPE_HEAD_DIM:]

    k = torch.empty(
        (num_tokens, NUM_LOCAL_HEADS, QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    return dict(k=k, k_nope=k_nope, k_rope=k_rope)


def fn_torch(k, k_nope, k_rope):
    """Native PyTorch implementation."""
    k[..., :QK_NOPE_HEAD_DIM] = k_nope
    k[..., QK_NOPE_HEAD_DIM:] = k_rope


@torch.compile(dynamic=True)
def fn_torch_compiled(k, k_nope, k_rope):
    """torch.compile optimized implementation."""
    return fn_torch(k, k_nope, k_rope)


def fn_flashinfer(k, k_nope, k_rope):
    """FlashInfer CUDA kernel implementation."""
    concat_mla_k_flashinfer(k, k_nope, k_rope)


def execute_and_get_output(f, data):
    """Execute function and return output for correctness checking."""
    data["k"].zero_()
    f(**data)
    assert data["k"].sum().item() != 0, "Output should not be all zeros"
    return data["k"].clone()


def verify_correctness():
    """Verify that all implementations produce the same output."""
    print("Verifying correctness...")
    torch.manual_seed(0)
    data = create_data(num_tokens=32768)

    output_ref = execute_and_get_output(fn_torch, data)
    output_flashinfer = execute_and_get_output(fn_flashinfer, data)

    if not torch.allclose(output_ref, output_flashinfer):
        abs_delta = torch.abs(output_ref - output_flashinfer)
        raise AssertionError(
            f"FlashInfer output mismatch! "
            f"abs_delta max={abs_delta.max().item()}, "
            f"num_mismatches={torch.sum(abs_delta != 0).item()}"
        )

    print("All implementations produce correct results!")


def benchmark():
    """Run benchmark for all implementations."""
    num_tokens_list = [2048, 4096, 8192, 16384, 32768]
    providers = [
        ("torch", fn_torch),
        ("torch_compiled", fn_torch_compiled),
        ("flashinfer", fn_flashinfer),
    ]

    # Warmup torch_compiled
    print("Warming up torch.compile...")
    data = create_data(num_tokens=2048)
    for _ in range(3):
        fn_torch_compiled(**data)

    # Collect results
    results = {name: [] for name, _ in providers}

    for num_tokens in num_tokens_list:
        data = create_data(num_tokens=num_tokens)
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
