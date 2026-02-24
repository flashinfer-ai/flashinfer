"""Test 1: CUDA vs Triton Philox randint — bitwise comparison.

Compiles a minimal CUDA kernel (via torch load_inline) that calls our
philox_randint from common.cuh. Compares output against a Triton kernel
that calls tl.randint with the same (seed, offset, n_rounds).
"""

import pathlib

import pytest
import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline


# ---------------------------------------------------------------------------
# Triton reference kernel: just tl.randint → store uint32
# ---------------------------------------------------------------------------
@triton.jit
def _triton_philox_kernel(
    out_ptr,
    seed_ptr,
    n_elements,
    N_ROUNDS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    seed = tl.load(seed_ptr)
    rand = tl.randint(seed, offsets, N_ROUNDS)
    # rand is uint32 but Triton stores it as int32 bit-pattern
    tl.store(out_ptr + offsets, rand, mask=mask)


def triton_philox(seed: int, n_elements: int, n_rounds: int) -> torch.Tensor:
    """Run the Triton kernel and return int32 tensor (uint32 bit-pattern)."""
    seed_t = torch.tensor([seed], dtype=torch.int64, device="cuda")
    out = torch.empty(n_elements, dtype=torch.int32, device="cuda")
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _triton_philox_kernel[grid](out, seed_t, n_elements, n_rounds, BLOCK_SIZE)
    return out


# ---------------------------------------------------------------------------
# CUDA kernel via load_inline
# ---------------------------------------------------------------------------
_FLASHINFER_INCLUDE = str(pathlib.Path(__file__).resolve().parents[2] / "include")

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <flashinfer/mamba/common.cuh>

__global__ void philox_kernel(int32_t* out, int64_t seed, int n_elements,
                              int n_rounds) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;
    uint32_t result = flashinfer::mamba::philox_randint(seed, (uint32_t)idx, n_rounds);
    out[idx] = static_cast<int32_t>(result);
}

torch::Tensor cuda_philox(int64_t seed, int n_elements, int n_rounds) {
    auto out = torch::empty({n_elements}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    int threads = 256;
    int blocks = (n_elements + threads - 1) / threads;
    philox_kernel<<<blocks, threads>>>(out.data_ptr<int32_t>(), seed, n_elements, n_rounds);
    return out;
}
"""

_CPP_SOURCE = r"""
torch::Tensor cuda_philox(int64_t seed, int n_elements, int n_rounds);
"""


@pytest.fixture(scope="module")
def cuda_philox_fn():
    """Compile the inline CUDA module once per test module."""
    mod = load_inline(
        name="test_philox_randint",
        cpp_sources=[_CPP_SOURCE],
        cuda_sources=[_CUDA_SOURCE],
        extra_include_paths=[_FLASHINFER_INCLUDE],
        functions=["cuda_philox"],
        verbose=False,
    )
    return mod.cuda_philox


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_rounds", [1, 4, 10])
@pytest.mark.parametrize("seed", [0, 42, 123456, 2**31 - 1])
def test_philox_randint_matches_triton(cuda_philox_fn, seed, n_rounds):
    """Bitwise comparison of CUDA philox_randint vs Triton tl.randint."""
    n_elements = 1024

    cuda_out = cuda_philox_fn(seed, n_elements, n_rounds)
    triton_out = triton_philox(seed, n_elements, n_rounds)

    mismatches = (cuda_out != triton_out).sum().item()
    if mismatches > 0:
        # Show first few mismatches for debugging
        diff_idx = torch.where(cuda_out != triton_out)[0][:10]
        for idx in diff_idx:
            i = idx.item()
            c = cuda_out[i].item() & 0xFFFFFFFF
            t = triton_out[i].item() & 0xFFFFFFFF
            print(f"  offset={i}: CUDA=0x{c:08X}, Triton=0x{t:08X}")

    assert mismatches == 0, (
        f"seed={seed}, n_rounds={n_rounds}: {mismatches}/{n_elements} mismatches"
    )
    print(f"  seed={seed}, n_rounds={n_rounds}: all {n_elements} values match")
