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
"""

_PHILOX_CPP_SOURCE = r"""
torch::Tensor cuda_philox(int64_t seed, int n_elements, int n_rounds);
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def philox_module():
    """Compile philox_randint test kernel (works on any GPU)."""
    return load_inline(
        name="test_philox",
        cpp_sources=[_PHILOX_CPP_SOURCE],
        cuda_sources=[_PHILOX_CUDA_SOURCE],
        extra_include_paths=[_FLASHINFER_INCLUDE],
        functions=["cuda_philox"],
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
