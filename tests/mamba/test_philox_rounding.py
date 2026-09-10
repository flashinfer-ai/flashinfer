"""Tests for Philox PRNG and stochastic rounding primitives.

Test 1: CUDA vs Triton Philox randint — bitwise comparison (any GPU).
Test 2: CUDA vs Triton stochastic rounding (cvt.rs.f16x2.f32) — bitwise comparison (sm_100a+).
"""

import pathlib

import pytest
import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline

from flashinfer.utils import get_compute_capability, is_cvt_rs_supported


# ---------------------------------------------------------------------------
# Triton reference: tl.randint
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
    """Run the Triton philox kernel and return int32 tensor (uint32 bit-pattern)."""
    seed_t = torch.tensor([seed], dtype=torch.int64, device="cuda")
    out = torch.empty(n_elements, dtype=torch.int32, device="cuda")
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _triton_philox_kernel[grid](out, seed_t, n_elements, n_rounds, BLOCK_SIZE)
    return out


# ---------------------------------------------------------------------------
# Triton reference: tl.randint4x with i64 offsets
# Triton's randint4x splits an i64 offset across Philox c0 (low 32 bits) and
# c1 (high 32 bits) — see triton/language/random.py:randint4x.  This is the
# behavior that the checkpointing_state_update kernel relies on (its
# `base_rand` is computed as i64 via `cache_batch_idx.to(tl.int64)`).
# ---------------------------------------------------------------------------
@triton.jit
def _triton_philox4x_offsets_kernel(
    out_ptr,
    seed_ptr,
    offsets_ptr,
    n_elements,
    N_ROUNDS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_offs < n_elements
    seed = tl.load(seed_ptr)
    offsets = tl.load(offsets_ptr + block_offs, mask=mask)
    r0, r1, r2, r3 = tl.randint4x(seed, offsets, N_ROUNDS)
    tl.store(out_ptr + block_offs * 4 + 0, r0, mask=mask)
    tl.store(out_ptr + block_offs * 4 + 1, r1, mask=mask)
    tl.store(out_ptr + block_offs * 4 + 2, r2, mask=mask)
    tl.store(out_ptr + block_offs * 4 + 3, r3, mask=mask)


def triton_philox_offsets(
    seed: int, offsets: torch.Tensor, n_rounds: int
) -> torch.Tensor:
    """Run Triton's tl.randint4x with explicit i64 offsets; returns (n, 4) int32."""
    assert offsets.dtype == torch.int64
    seed_t = torch.tensor([seed], dtype=torch.int64, device="cuda")
    n = offsets.numel()
    out = torch.empty((n, 4), dtype=torch.int32, device="cuda")
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _triton_philox4x_offsets_kernel[grid](out, seed_t, offsets, n, n_rounds, BLOCK_SIZE)
    return out


# ---------------------------------------------------------------------------
# Triton reference: convert_rs_fp16x2 (stochastic rounding)
# ---------------------------------------------------------------------------
@triton.jit
def _triton_convert_rs_kernel(
    out_ptr,
    fp32_ptr,
    rand_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply stochastic rounding: fp32 → fp16 using random bits."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(fp32_ptr + offsets, mask=mask)
    rand = tl.load(rand_ptr + offsets, mask=mask)
    # cvt.rs.f16x2.f32: stochastic rounding of fp32 pair → fp16x2
    y = tl.inline_asm_elementwise(
        asm="""{
        cvt.rs.f16x2.f32 $0, $2, $1, $3;
        }""",
        constraints=("=r,r,r,r,r"),
        args=(x, rand),
        dtype=tl.float16,
        is_pure=True,
        pack=2,
    )
    tl.store(out_ptr + offsets, y, mask=mask)


def triton_stochastic_round(
    fp32_values: torch.Tensor, rand_bits: torch.Tensor
) -> torch.Tensor:
    """Stochastic-round fp32 → fp16 using random bits via Triton PTX."""
    assert fp32_values.dtype == torch.float32
    assert rand_bits.dtype == torch.int32
    n = fp32_values.numel()
    # n must be even for fp16x2 packing
    assert n % 2 == 0, "n_elements must be even for fp16x2 packing"
    out = torch.empty(n, dtype=torch.float16, device="cuda")
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _triton_convert_rs_kernel[grid](out, fp32_values, rand_bits, n, BLOCK_SIZE)
    return out


# ---------------------------------------------------------------------------
# Triton reference: cvt.rs.satfinite.e4m3x4.f32 (stochastic rounding fp32 → fp8 e4m3)
# ---------------------------------------------------------------------------
@triton.jit
def _triton_convert_rs_e4m3_kernel(
    out_ptr,
    fp32_ptr,
    rand_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply stochastic rounding: fp32 → fp8 e4m3 using random bits.

    PTX cvt.rs.satfinite.e4m3x4.f32 packs 4 fp8 outputs into a 32-bit
    register; the reversed source-register order {$4,$3,$2,$1} is
    load-bearing — see _stochastic_round_fp8x4_e4m3 in
    triton_reference/replay_selective_state_update.py for the full rationale.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(fp32_ptr + offsets, mask=mask)
    rand = tl.load(rand_ptr + offsets, mask=mask)
    y = tl.inline_asm_elementwise(
        asm="cvt.rs.satfinite.e4m3x4.f32 $0, {$4, $3, $2, $1}, $5;",
        constraints="=r,r,r,r,r,r,r,r,r",
        args=(x, rand),
        dtype=tl.float8e4nv,
        is_pure=True,
        pack=4,
    )
    tl.store(out_ptr + offsets, y, mask=mask)


def triton_stochastic_round_e4m3(
    fp32_values: torch.Tensor, rand_bits: torch.Tensor
) -> torch.Tensor:
    """Stochastic-round fp32 → fp8 e4m3 using random bits via Triton PTX."""
    assert fp32_values.dtype == torch.float32
    assert rand_bits.dtype == torch.int32
    n = fp32_values.numel()
    # n must be a multiple of 4 for fp8x4 packing
    assert n % 4 == 0, "n_elements must be a multiple of 4 for fp8x4 packing"
    out = torch.empty(n, dtype=torch.float8_e4m3fn, device="cuda")
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _triton_convert_rs_e4m3_kernel[grid](out, fp32_values, rand_bits, n, BLOCK_SIZE)
    return out


# ---------------------------------------------------------------------------
# CUDA sources
# ---------------------------------------------------------------------------
_FLASHINFER_INCLUDE = str(pathlib.Path(__file__).resolve().parents[2] / "include")

_PHILOX_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <flashinfer/mamba/conversion.cuh>

template <int N_ROUNDS>
__global__ void philox_kernel(int32_t* out, int64_t seed, int n_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;
    uint32_t result = flashinfer::mamba::conversion::philox_randint<N_ROUNDS>(seed, (uint32_t)idx);
    out[idx] = static_cast<int32_t>(result);
}

torch::Tensor cuda_philox(int64_t seed, int n_elements, int n_rounds) {
    auto out = torch::empty({n_elements}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    int threads = 256;
    int blocks = (n_elements + threads - 1) / threads;
    switch (n_rounds) {
        case 1:  philox_kernel<1><<<blocks, threads>>>(out.data_ptr<int32_t>(), seed, n_elements); break;
        case 4:  philox_kernel<4><<<blocks, threads>>>(out.data_ptr<int32_t>(), seed, n_elements); break;
        case 10: philox_kernel<10><<<blocks, threads>>>(out.data_ptr<int32_t>(), seed, n_elements); break;
        default: TORCH_CHECK(false, "Unsupported n_rounds: ", n_rounds);
    }
    return out;
}

// 4x variant with explicit i64 offsets — exercises the high-bit path of
// Triton's tl.randint4x (which splits an i64 offset across Philox c0/c1).
template <int N_ROUNDS>
__global__ void philox_offsets_kernel(int32_t* out, int64_t seed,
                                      const int64_t* offsets, int n_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;
    uint32_t r0, r1, r2, r3;
    flashinfer::mamba::conversion::philox_randint4x<N_ROUNDS>(
        seed, offsets[idx], r0, r1, r2, r3);
    out[idx * 4 + 0] = static_cast<int32_t>(r0);
    out[idx * 4 + 1] = static_cast<int32_t>(r1);
    out[idx * 4 + 2] = static_cast<int32_t>(r2);
    out[idx * 4 + 3] = static_cast<int32_t>(r3);
}

torch::Tensor cuda_philox_offsets(int64_t seed, torch::Tensor offsets, int n_rounds) {
    TORCH_CHECK(offsets.dtype() == torch::kInt64, "offsets must be int64");
    TORCH_CHECK(offsets.is_cuda(), "offsets must be CUDA tensor");
    int n = offsets.numel();
    auto out = torch::empty({n, 4}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    switch (n_rounds) {
        case 1:  philox_offsets_kernel<1><<<blocks, threads>>>(out.data_ptr<int32_t>(),  seed, offsets.data_ptr<int64_t>(), n); break;
        case 4:  philox_offsets_kernel<4><<<blocks, threads>>>(out.data_ptr<int32_t>(),  seed, offsets.data_ptr<int64_t>(), n); break;
        case 10: philox_offsets_kernel<10><<<blocks, threads>>>(out.data_ptr<int32_t>(), seed, offsets.data_ptr<int64_t>(), n); break;
        default: TORCH_CHECK(false, "Unsupported n_rounds: ", n_rounds);
    }
    return out;
}
"""

_PHILOX_CPP_SOURCE = r"""
torch::Tensor cuda_philox(int64_t seed, int n_elements, int n_rounds);
torch::Tensor cuda_philox_offsets(int64_t seed, torch::Tensor offsets, int n_rounds);
"""

_STOCHASTIC_ROUND_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <flashinfer/mamba/conversion.cuh>

__global__ void stochastic_round_kernel(half* out, const float* fp32_in,
                                        const int32_t* rand_in, int n_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pair_idx = idx * 2;
    if (pair_idx + 1 >= n_elements) return;

    float a = fp32_in[pair_idx];
    float b = fp32_in[pair_idx + 1];
    // Triton pack=2 uses rand from the first element of the pair
    uint32_t rand = *reinterpret_cast<const uint32_t*>(&rand_in[pair_idx]);

    uint32_t packed = flashinfer::mamba::conversion::cvt_rs_f16x2_f32(a, b, rand);
    *reinterpret_cast<uint32_t*>(&out[pair_idx]) = packed;
}

torch::Tensor cuda_stochastic_round(torch::Tensor fp32_values, torch::Tensor rand_bits) {
    int n = fp32_values.numel();
    auto out = torch::empty({n}, torch::dtype(torch::kFloat16).device(torch::kCUDA));
    int n_pairs = n / 2;
    int threads = 256;
    int blocks = (n_pairs + threads - 1) / threads;
    stochastic_round_kernel<<<blocks, threads>>>(
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        fp32_values.data_ptr<float>(),
        rand_bits.data_ptr<int32_t>(),
        n);
    return out;
}
"""

_STOCHASTIC_ROUND_CPP_SOURCE = r"""
torch::Tensor cuda_stochastic_round(torch::Tensor fp32_values, torch::Tensor rand_bits);
"""

_STOCHASTIC_ROUND_SINGLE_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <flashinfer/mamba/conversion.cuh>

// Each thread converts one fp32 value to fp16 using cvt_rs_f16_f32 (single-value).
// rand13_in contains 13-bit random values (one per element, stored as int32).
__global__ void stochastic_round_single_kernel(half* out, const float* fp32_in,
                                                const int32_t* rand13_in, int n_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    float x = fp32_in[idx];
    uint32_t rand13 = static_cast<uint32_t>(rand13_in[idx]) & 0x1FFFu;
    out[idx] = flashinfer::mamba::conversion::cvt_rs_f16_f32(x, rand13);
}

torch::Tensor cuda_stochastic_round_single(torch::Tensor fp32_values, torch::Tensor rand13_bits) {
    int n = fp32_values.numel();
    auto out = torch::empty({n}, torch::dtype(torch::kFloat16).device(torch::kCUDA));
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    stochastic_round_single_kernel<<<blocks, threads>>>(
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        fp32_values.data_ptr<float>(),
        rand13_bits.data_ptr<int32_t>(),
        n);
    return out;
}
"""

_STOCHASTIC_ROUND_SINGLE_CPP_SOURCE = r"""
torch::Tensor cuda_stochastic_round_single(torch::Tensor fp32_values, torch::Tensor rand13_bits);
"""

_STOCHASTIC_ROUND_E4M3_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <flashinfer/mamba/conversion.cuh>

// Each thread converts 4 fp32 values to fp8 e4m3 (packed in a uint32).
// Triton pack=4 takes the first rand of every 4-element group; we mirror that.
__global__ void stochastic_round_e4m3_kernel(uint8_t* out, const float* fp32_in,
                                             const int32_t* rand_in, int n_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int quad_idx = idx * 4;
    if (quad_idx + 3 >= n_elements) return;

    float a = fp32_in[quad_idx + 0];
    float b = fp32_in[quad_idx + 1];
    float c = fp32_in[quad_idx + 2];
    float d = fp32_in[quad_idx + 3];
    uint32_t rbits = *reinterpret_cast<const uint32_t*>(&rand_in[quad_idx]);

    uint32_t packed = flashinfer::mamba::conversion::cvt_rs_e4m3x4_f32(a, b, c, d, rbits);
    *reinterpret_cast<uint32_t*>(&out[quad_idx]) = packed;
}

torch::Tensor cuda_stochastic_round_e4m3(torch::Tensor fp32_values, torch::Tensor rand_bits) {
    int n = fp32_values.numel();
    auto out = torch::empty({n}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
    int n_quads = n / 4;
    int threads = 256;
    int blocks = (n_quads + threads - 1) / threads;
    stochastic_round_e4m3_kernel<<<blocks, threads>>>(
        out.data_ptr<uint8_t>(),
        fp32_values.data_ptr<float>(),
        rand_bits.data_ptr<int32_t>(),
        n);
    return out;
}
"""

_STOCHASTIC_ROUND_E4M3_CPP_SOURCE = r"""
torch::Tensor cuda_stochastic_round_e4m3(torch::Tensor fp32_values, torch::Tensor rand_bits);
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def philox_module():
    """Compile philox_randint test kernels (works on any GPU)."""
    return load_inline(
        name="test_philox",
        cpp_sources=[_PHILOX_CPP_SOURCE],
        cuda_sources=[_PHILOX_CUDA_SOURCE],
        extra_include_paths=[_FLASHINFER_INCLUDE],
        functions=["cuda_philox", "cuda_philox_offsets"],
        verbose=False,
    )


@pytest.fixture(scope="module")
def stochastic_round_module():
    """Compile cvt_rs_f16x2_f32 test kernel with sm_100a (hardware PTX path)."""
    major, minor = get_compute_capability(torch.device("cuda"))
    if not is_cvt_rs_supported(torch.device("cuda")):
        pytest.skip("cvt.rs.f16x2.f32 requires sm_100a; not supported on this GPU")
    # Append 'a' suffix for SM >= 9, matching flashinfer/compilation_context.py:44-45
    minor_str = f"{minor}a" if major >= 9 else str(minor)
    gencode = f"-gencode=arch=compute_{major}{minor_str},code=sm_{major}{minor_str}"
    return load_inline(
        name="test_stochastic_round",
        cpp_sources=[_STOCHASTIC_ROUND_CPP_SOURCE],
        cuda_sources=[_STOCHASTIC_ROUND_CUDA_SOURCE],
        extra_include_paths=[_FLASHINFER_INCLUDE],
        extra_cuda_cflags=[gencode],
        functions=["cuda_stochastic_round"],
        verbose=False,
    )


@pytest.fixture(scope="module")
def stochastic_round_sw_module():
    """Compile cvt_rs_f16x2_f32 test kernel without sm_100a (software fallback path)."""
    return load_inline(
        name="test_stochastic_round_sw",
        cpp_sources=[_STOCHASTIC_ROUND_CPP_SOURCE],
        cuda_sources=[_STOCHASTIC_ROUND_CUDA_SOURCE],
        extra_include_paths=[_FLASHINFER_INCLUDE],
        functions=["cuda_stochastic_round"],
        verbose=False,
    )


@pytest.fixture(scope="module")
def stochastic_round_single_module():
    """Compile cvt_rs_f16_f32 single-value test kernel (software path, any GPU)."""
    return load_inline(
        name="test_stochastic_round_single",
        cpp_sources=[_STOCHASTIC_ROUND_SINGLE_CPP_SOURCE],
        cuda_sources=[_STOCHASTIC_ROUND_SINGLE_CUDA_SOURCE],
        extra_include_paths=[_FLASHINFER_INCLUDE],
        functions=["cuda_stochastic_round_single"],
        verbose=False,
    )


@pytest.fixture(scope="module")
def stochastic_round_e4m3_module():
    """Compile cvt_rs_e4m3x4_f32 test kernel with sm_100a (hardware PTX path)."""
    major, minor = get_compute_capability(torch.device("cuda"))
    if not is_cvt_rs_supported(torch.device("cuda")):
        pytest.skip(
            "cvt.rs.satfinite.e4m3x4.f32 requires sm_100a; not supported on this GPU"
        )
    minor_str = f"{minor}a" if major >= 9 else str(minor)
    gencode = f"-gencode=arch=compute_{major}{minor_str},code=sm_{major}{minor_str}"
    return load_inline(
        name="test_stochastic_round_e4m3",
        cpp_sources=[_STOCHASTIC_ROUND_E4M3_CPP_SOURCE],
        cuda_sources=[_STOCHASTIC_ROUND_E4M3_CUDA_SOURCE],
        extra_include_paths=[_FLASHINFER_INCLUDE],
        extra_cuda_cflags=[gencode],
        functions=["cuda_stochastic_round_e4m3"],
        verbose=False,
    )


@pytest.fixture(scope="module")
def stochastic_round_e4m3_sw_module():
    """Compile cvt_rs_e4m3x4_f32 test kernel WITHOUT sm_100a (software fallback path)."""
    return load_inline(
        name="test_stochastic_round_e4m3_sw",
        cpp_sources=[_STOCHASTIC_ROUND_E4M3_CPP_SOURCE],
        cuda_sources=[_STOCHASTIC_ROUND_E4M3_CUDA_SOURCE],
        extra_include_paths=[_FLASHINFER_INCLUDE],
        functions=["cuda_stochastic_round_e4m3"],
        verbose=False,
    )


# ---------------------------------------------------------------------------
# Test 1: Philox randint (any GPU)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_rounds", [1, 4, 10])
@pytest.mark.parametrize("seed", [0, 42, 123456, 2**31 - 1])
def test_philox_randint(philox_module, seed, n_rounds):
    """Bitwise comparison of CUDA philox_randint vs Triton tl.randint."""
    n_elements = 1024

    cuda_out = philox_module.cuda_philox(seed, n_elements, n_rounds)
    triton_out = triton_philox(seed, n_elements, n_rounds)

    mismatches = (cuda_out != triton_out).sum().item()
    if mismatches > 0:
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


# ---------------------------------------------------------------------------
# Test 1b: Philox randint4x with i64 offsets > 2^32 (any GPU)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_rounds", [10])
@pytest.mark.parametrize("seed", [0, 42, 0xDEADBEEF])
def test_philox_randint_large_offsets(philox_module, seed, n_rounds):
    """Bitwise comparison of CUDA philox_randint4x vs Triton tl.randint4x for
    offsets that exceed 2^32 — verifies the i64 counter split (c0=low, c1=high)
    matches Triton.  checkpointing_state_update computes `base_rand =
    cache_batch_idx * stride_state_batch + ...` as i64 (cache_batch_idx is
    .to(int64)), so c1 != 0 for offsets >= 2^32."""
    # All offsets >= 2^32 to exercise the high-bit (c1) path.
    offsets_boundary = torch.tensor(
        [
            0x100000000,  # 2^32 exactly
            0x100000001,  # 2^32 + 1
            0x123456789ABCDEF0,  # large 64-bit value
            0x7FFFFFFFFFFFFFFF,  # max int64
        ],
        dtype=torch.int64,
        device="cuda",
    )
    torch.manual_seed(seed)
    offsets_random = torch.randint(
        0x100000000,
        0x7FFFFFFFFFFFFFFF,
        (256,),
        dtype=torch.int64,
        device="cuda",
    )
    offsets = torch.cat([offsets_boundary, offsets_random])
    n = offsets.numel()

    cuda_out = philox_module.cuda_philox_offsets(seed, offsets, n_rounds)
    triton_out = triton_philox_offsets(seed, offsets, n_rounds)

    diff = (cuda_out != triton_out).any(dim=1)
    mismatches = diff.sum().item()
    if mismatches > 0:
        diff_idx = torch.where(diff)[0][:10]
        for idx in diff_idx:
            i = idx.item()
            off = offsets[i].item()
            c0 = off & 0xFFFFFFFF
            c1 = (off >> 32) & 0xFFFFFFFF
            c = [cuda_out[i, j].item() & 0xFFFFFFFF for j in range(4)]
            t = [triton_out[i, j].item() & 0xFFFFFFFF for j in range(4)]
            print(
                f"  offset=0x{off:016X} (c0=0x{c0:08X}, c1=0x{c1:08X}):\n"
                f"    CUDA:   {[f'0x{v:08X}' for v in c]}\n"
                f"    Triton: {[f'0x{v:08X}' for v in t]}"
            )

    assert mismatches == 0, (
        f"seed={seed}, n_rounds={n_rounds}: {mismatches}/{n} mismatches"
    )
    print(f"  seed={seed}, n_rounds={n_rounds}: all {n} 4-tuples match")


# ---------------------------------------------------------------------------
# Test 2: Stochastic rounding (sm_100a+ only)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 42, 99999])
def test_stochastic_rounding(stochastic_round_module, seed):
    """Bitwise comparison of CUDA vs Triton stochastic rounding (cvt.rs.f16x2.f32)."""
    n_elements = 1024  # must be even
    torch.manual_seed(seed)

    fp32_values = torch.randn(n_elements, dtype=torch.float32, device="cuda")
    rand_bits = torch.randint(
        -(2**31), 2**31, (n_elements,), dtype=torch.int32, device="cuda"
    )

    cuda_out = stochastic_round_module.cuda_stochastic_round(fp32_values, rand_bits)
    triton_out = triton_stochastic_round(fp32_values, rand_bits)

    # Compare as raw uint16 bit patterns
    cuda_bits = cuda_out.view(torch.int16)
    triton_bits = triton_out.view(torch.int16)

    mismatches = (cuda_bits != triton_bits).sum().item()
    if mismatches > 0:
        diff_idx = torch.where(cuda_bits != triton_bits)[0][:10]
        for idx in diff_idx:
            i = idx.item()
            pair = i // 2
            cb = cuda_bits[i].item() & 0xFFFF
            tb = triton_bits[i].item() & 0xFFFF
            cf = cuda_out[i].item()
            tf = triton_out[i].item()
            rb = rand_bits[pair].item() & 0xFFFFFFFF
            print(
                f"  elem={i} (pair={pair}): fp32={fp32_values[i].item():.6f}, "
                f"rand=0x{rb:08X}, CUDA=0x{cb:04X}({cf}), Triton=0x{tb:04X}({tf})"
            )

    assert mismatches == 0, f"seed={seed}: {mismatches}/{n_elements} mismatches"
    print(f"  seed={seed}: all {n_elements} fp16 values match bitwise (hw)")


# ---------------------------------------------------------------------------
# Test 3: Stochastic rounding software fallback (any GPU, verified on Blackwell)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 42, 99999])
def test_stochastic_rounding_sw(
    stochastic_round_sw_module, stochastic_round_module, seed
):
    """Software stochastic rounding matches hardware PTX path bitwise."""
    n_elements = 1024  # must be even
    torch.manual_seed(seed)

    fp32_values = torch.randn(n_elements, dtype=torch.float32, device="cuda")
    rand_bits = torch.randint(
        -(2**31), 2**31, (n_elements,), dtype=torch.int32, device="cuda"
    )

    sw_out = stochastic_round_sw_module.cuda_stochastic_round(fp32_values, rand_bits)
    hw_out = stochastic_round_module.cuda_stochastic_round(fp32_values, rand_bits)

    # Compare as raw uint16 bit patterns
    sw_bits = sw_out.view(torch.int16)
    hw_bits = hw_out.view(torch.int16)

    mismatches = (sw_bits != hw_bits).sum().item()
    if mismatches > 0:
        diff_idx = torch.where(sw_bits != hw_bits)[0][:10]
        for idx in diff_idx:
            i = idx.item()
            pair = i // 2
            sb = sw_bits[i].item() & 0xFFFF
            hb = hw_bits[i].item() & 0xFFFF
            sf = sw_out[i].item()
            hf = hw_out[i].item()
            rb = rand_bits[pair].item() & 0xFFFFFFFF
            print(
                f"  elem={i} (pair={pair}): fp32={fp32_values[i].item():.6f}, "
                f"rand=0x{rb:08X}, SW=0x{sb:04X}({sf}), HW=0x{hb:04X}({hf})"
            )

    assert mismatches == 0, f"seed={seed}: {mismatches}/{n_elements} mismatches"
    print(f"  seed={seed}: all {n_elements} fp16 values match bitwise (sw vs hw)")


# ---------------------------------------------------------------------------
# Test 4: Single-value stochastic rounding matches pair-wise (any GPU)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 42, 99999])
def test_stochastic_rounding_single_vs_pair(
    stochastic_round_single_module, stochastic_round_sw_module, seed
):
    """cvt_rs_f16_f32 (single) matches the corresponding element from cvt_rs_f16x2_f32 (pair)."""
    n_elements = 1024  # must be even
    torch.manual_seed(seed)

    fp32_values = torch.randn(n_elements, dtype=torch.float32, device="cuda")
    # Generate 13-bit random values per element
    rand13 = torch.randint(0, 8192, (n_elements,), dtype=torch.int32, device="cuda")

    # Single-value path: cvt_rs_f16_f32(x, rand13) for each element
    single_out = stochastic_round_single_module.cuda_stochastic_round_single(
        fp32_values, rand13
    )

    # Pair-wise path: cvt_rs_f16x2_f32(a, b, rbits) where rbits packs rand13 for both
    # rbits layout: bits[12:0] = rand for C++ a (low half), bits[28:16] = rand for C++ b (high half)
    rand_a = rand13[0::2]  # even elements
    rand_b = rand13[1::2]  # odd elements
    rbits = (rand_a & 0x1FFF) | ((rand_b & 0x1FFF) << 16)
    # Expand rbits back to n_elements (pair-wise kernel reads from pair_idx)
    rbits_expanded = torch.zeros(n_elements, dtype=torch.int32, device="cuda")
    rbits_expanded[0::2] = rbits
    rbits_expanded[1::2] = rbits  # pair kernel reads from pair_idx = even index

    pair_out = stochastic_round_sw_module.cuda_stochastic_round(
        fp32_values, rbits_expanded
    )

    # Compare as raw bit patterns
    single_bits = single_out.view(torch.int16)
    pair_bits = pair_out.view(torch.int16)

    mismatches = (single_bits != pair_bits).sum().item()
    if mismatches > 0:
        diff_idx = torch.where(single_bits != pair_bits)[0][:10]
        for idx in diff_idx:
            i = idx.item()
            sb = single_bits[i].item() & 0xFFFF
            pb = pair_bits[i].item() & 0xFFFF
            sf = single_out[i].item()
            pf = pair_out[i].item()
            r13 = rand13[i].item() & 0x1FFF
            print(
                f"  elem={i}: fp32={fp32_values[i].item():.6f}, "
                f"rand13=0x{r13:04X}, single=0x{sb:04X}({sf}), pair=0x{pb:04X}({pf})"
            )

    assert mismatches == 0, f"seed={seed}: {mismatches}/{n_elements} mismatches"
    print(f"  seed={seed}: all {n_elements} fp16 values match bitwise (single vs pair)")


# ---------------------------------------------------------------------------
# Test 5: e4m3 stochastic rounding (sm_100a+ only)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 42, 99999])
def test_stochastic_rounding_e4m3(stochastic_round_e4m3_module, seed):
    """Bitwise comparison of CUDA vs Triton stochastic rounding (cvt.rs.satfinite.e4m3x4.f32)."""
    n_elements = 1024  # multiple of 4 for fp8x4 packing
    torch.manual_seed(seed)

    fp32_values = torch.randn(n_elements, dtype=torch.float32, device="cuda")
    # int32 for raw random bits — PTX takes the bit pattern, sign interpretation
    # doesn't matter.  Matches test_sr_grid_bracket in upstream
    # test_checkpointing_state_update.py.
    rand_bits = torch.randint(
        -(2**31), 2**31, (n_elements,), dtype=torch.int32, device="cuda"
    )

    cuda_out = stochastic_round_e4m3_module.cuda_stochastic_round_e4m3(
        fp32_values, rand_bits
    )
    triton_out = triton_stochastic_round_e4m3(fp32_values, rand_bits)

    # Compare as raw uint8 bit patterns (each fp8 is exactly 1 byte).
    cuda_bits = cuda_out  # already uint8
    triton_bits = triton_out.view(torch.uint8)

    mismatches = (cuda_bits != triton_bits).sum().item()
    if mismatches > 0:
        diff_idx = torch.where(cuda_bits != triton_bits)[0][:10]
        for idx in diff_idx:
            i = idx.item()
            quad = i // 4
            cb = cuda_bits[i].item()
            tb = triton_bits[i].item()
            rb = rand_bits[quad * 4].item() & 0xFFFFFFFF
            print(
                f"  elem={i} (quad={quad}): fp32={fp32_values[i].item():.6f}, "
                f"rand=0x{rb:08X}, CUDA=0x{cb:02X}, Triton=0x{tb:02X}"
            )

    assert mismatches == 0, f"seed={seed}: {mismatches}/{n_elements} mismatches"
    print(f"  seed={seed}: all {n_elements} fp8 values match bitwise (hw)")


# ---------------------------------------------------------------------------
# Test 6: e4m3 SW fallback matches HW bitwise (sm_100a+ for HW oracle).
# ---------------------------------------------------------------------------
def _make_e4m3_test_inputs(n: int, seed: int) -> torch.Tensor:
    """Generate fp32 inputs covering subnormal, normal, and saturation regions
    of e4m3 (FN, max finite = 448, smallest subnormal = 2^-9).  n must be a
    multiple of 6 * 4 (6 bands, fp8x4 packing)."""
    assert n % 24 == 0, "n must be a multiple of 24 for 6 bands × fp8x4 packing"
    torch.manual_seed(seed)
    nb = n // 6
    # First band: half uniform small values (covers e4m3 subnormal region),
    # half log-spaced tiny-normal fp32 in [1e-38, 1e-10] with random signs.
    # The log-spaced half exercises the flush-to-zero boundary (fp32 unbiased
    # < -49 ⇒ shift_truncate >= 64 in cvt_rs_e4m3_sw); without it the e4m3
    # SR path's tiny-normal UB guard wouldn't be exercised by this test.
    nb_half = nb // 2
    tiny_log = torch.logspace(
        -38.0, -10.0, nb - nb_half, base=10.0, dtype=torch.float32
    )
    tiny_sign = torch.where(torch.rand(nb - nb_half) < 0.5, -1.0, 1.0).to(torch.float32)
    tiny_band = torch.cat(
        [
            torch.empty(nb_half, dtype=torch.float32).uniform_(-0.02, 0.02),
            tiny_log * tiny_sign,
        ]
    )
    bands = [
        # Tiny / subnormal + log-spaced tiny-normal (flush-to-zero coverage)
        tiny_band,
        # Normal small
        torch.empty(nb, dtype=torch.float32).uniform_(-1.0, 1.0),
        # Normal medium
        torch.empty(nb, dtype=torch.float32).uniform_(-100.0, 100.0),
        # Near max finite (in-range)
        torch.empty(nb, dtype=torch.float32).uniform_(-448.0, 448.0),
        # Saturation overflow (1.5x max finite)
        torch.empty(nb, dtype=torch.float32).uniform_(-672.0, 672.0),
        # Random Gaussians (broad coverage of typical activation magnitudes)
        torch.randn(nb, dtype=torch.float32) * 50.0,
    ]
    return torch.cat(bands).to("cuda")


@pytest.mark.parametrize("seed", [0, 42, 99999])
def test_stochastic_rounding_e4m3_sw(
    stochastic_round_e4m3_sw_module, stochastic_round_e4m3_module, seed
):
    """Software e4m3 SR matches hardware PTX path bitwise across a comprehensive
    input set covering subnormal, normal, and saturation regions.

    Compares 65536 fp32 inputs (6 bands × ~10K each) against the HW oracle.
    Inputs intentionally exclude NaN/±Inf — we test those separately if needed.
    """
    n_elements = 65520  # 65520 = 24 × 2730 ⇒ multiple of 24
    fp32_values = _make_e4m3_test_inputs(n_elements, seed)
    rand_bits = torch.randint(
        -(2**31), 2**31, (n_elements,), dtype=torch.int32, device="cuda"
    )

    sw_out = stochastic_round_e4m3_sw_module.cuda_stochastic_round_e4m3(
        fp32_values, rand_bits
    )
    hw_out = stochastic_round_e4m3_module.cuda_stochastic_round_e4m3(
        fp32_values, rand_bits
    )

    # Compare as raw uint8 bit patterns.
    mismatches = (sw_out != hw_out).sum().item()
    if mismatches > 0:
        diff_idx = torch.where(sw_out != hw_out)[0][:20]
        for idx in diff_idx:
            i = idx.item()
            quad = i // 4
            within_quad = i % 4
            sb = sw_out[i].item()
            hb = hw_out[i].item()
            rb = rand_bits[quad * 4].item() & 0xFFFFFFFF
            # The 8-bit slice of rbits used for this element (per our SW slicing).
            slice8 = (rb >> (within_quad * 8)) & 0xFF
            print(
                f"  elem={i} (quad={quad}, lane={within_quad}): "
                f"fp32={fp32_values[i].item():+.6e}, "
                f"rand=0x{rb:08X} (slice[{within_quad}]=0x{slice8:02X}), "
                f"SW=0x{sb:02X}, HW=0x{hb:02X}"
            )

    assert mismatches == 0, (
        f"seed={seed}: {mismatches}/{n_elements} mismatches between SW and HW e4m3 SR"
    )
    print(f"  seed={seed}: all {n_elements} fp8 values match bitwise (sw vs hw)")
