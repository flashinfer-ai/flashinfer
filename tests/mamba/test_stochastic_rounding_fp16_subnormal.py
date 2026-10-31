"""Tests for the software fp32 -> fp16 stochastic rounding (cvt_rs_f16_sw) over the full fp16 range.

The software path is used wherever PTX ``cvt.rs`` is unavailable (e.g. sm_80, sm_90, sm_120).  It must
be unbiased (E[result] == x) everywhere, including the fp16 subnormal range |x| < 2^-14, and must stay
bit-identical to the previous add-and-truncate formulation (and to the hardware) for fp16-normal
results.  Every test enumerates all 2^13 values of the 13 random bits, so expectations are exact.

Test 1 (any GPU): unbiasedness across magnitudes 1e-8 ... 1e3, both signs.
Test 2 (any GPU): no flush to zero for fp16-subnormal inputs.
Test 3 (any GPU): bit-identity with the previous formulation for finite |x| >= 2^-14 (and NaN/overflow).
Test 3b (any GPU): +-Inf stays +-Inf for every random draw (the previous formulation added the noise to
        the Inf bit pattern and returned NaN for 8191 of 8192 draws).
Test 4 (cvt.rs GPUs): hardware cvt.rs.f16x2.f32 vs software — bitwise for fp16-normal inputs,
        and same expected value for fp16-subnormal inputs.
"""

import pathlib

import pytest
import torch
from torch.utils.cpp_extension import load_inline

from flashinfer.utils import get_compute_capability, is_cvt_rs_supported

_FLASHINFER_INCLUDE = str(pathlib.Path(__file__).resolve().parents[2] / "include")
_N_RAND = 8192  # all values of the 13 random bits

_FP16_MIN_NORMAL = 2.0**-14
_FP16_MIN_SUBNORMAL = 2.0**-24

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <flashinfer/mamba/conversion.cuh>

// The previous software formulation (add 13 random bits at fp32 mantissa [12:0], then truncate),
// kept here verbatim as the reference for fp16-normal results.
__device__ __forceinline__ uint16_t cvt_rs_f16_sw_reference(float x, uint32_t rand13) {
  uint32_t bits = __float_as_uint(x);
  uint32_t sign = bits & 0x80000000u;
  uint32_t abs_bits = bits & 0x7FFFFFFFu;
  abs_bits += (rand13 & 0x1FFFu);
  uint32_t f32_exp = (abs_bits >> 23) & 0xFFu;
  uint32_t f32_mantissa = abs_bits & 0x7FFFFFu;
  uint16_t f16_bits;
  if (f32_exp == 0xFF) {
    f16_bits = (f32_mantissa != 0) ? 0x7E00u : 0x7C00u;
  } else if (f32_exp > 142) {
    f16_bits = 0x7C00u;
  } else if (f32_exp < 113) {
    f16_bits = 0;
  } else {
    uint16_t f16_exp = static_cast<uint16_t>(f32_exp - 112);
    uint16_t f16_mantissa = static_cast<uint16_t>(f32_mantissa >> 13);
    f16_bits = (f16_exp << 10) | f16_mantissa;
  }
  return static_cast<uint16_t>(sign >> 16) | f16_bits;
}

// out[i, r] = convert(x[i], r) for r in [0, 8192).  mode 0: cvt_rs_f16_sw, mode 1: reference,
// mode 2: cvt_rs_f16_f32 (hardware cvt.rs when compiled for a cvt.rs-capable "a" target).
__global__ void sr_all_rand_kernel(int16_t* out, const float* x, int n, int mode) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n * 8192) return;
  int i = idx / 8192;
  uint32_t r = static_cast<uint32_t>(idx % 8192);
  uint16_t h;
  if (mode == 0) {
    h = flashinfer::mamba::conversion::cvt_rs_f16_sw(x[i], r);
  } else if (mode == 1) {
    h = cvt_rs_f16_sw_reference(x[i], r);
  } else {
    h = __half_as_ushort(flashinfer::mamba::conversion::cvt_rs_f16_f32(x[i], r));
  }
  out[idx] = static_cast<int16_t>(h);
}

torch::Tensor sr_all_rand(torch::Tensor x, int64_t mode) {
  TORCH_CHECK(x.dtype() == torch::kFloat32 && x.is_cuda(), "x must be a CUDA float32 tensor");
  int n = x.numel();
  auto out = torch::empty({n, 8192}, torch::dtype(torch::kInt16).device(torch::kCUDA));
  int threads = 256;
  int blocks = (n * 8192 + threads - 1) / threads;
  sr_all_rand_kernel<<<blocks, threads>>>(out.data_ptr<int16_t>(), x.data_ptr<float>(), n,
                                          static_cast<int>(mode));
  return out;
}
"""

_CPP_SOURCE = r"""
torch::Tensor sr_all_rand(torch::Tensor x, int64_t mode);
"""


@pytest.fixture(scope="module")
def sr_module():
    """Software path + reference (default arch flags: no cvt.rs)."""
    return load_inline(
        name="test_sr_fp16_subnormal",
        cpp_sources=[_CPP_SOURCE],
        cuda_sources=[_CUDA_SOURCE],
        extra_include_paths=[_FLASHINFER_INCLUDE],
        functions=["sr_all_rand"],
        verbose=False,
    )


@pytest.fixture(scope="module")
def sr_hw_module():
    """Same kernel compiled for the native cvt.rs target (mode 2 then uses hardware cvt.rs)."""
    if not is_cvt_rs_supported(torch.device("cuda")):
        pytest.skip(
            "hardware stochastic rounding (cvt.rs) is not available on this GPU"
        )
    major, minor = get_compute_capability(torch.device("cuda"))
    minor_str = f"{minor}a" if major >= 9 else str(minor)
    gencode = f"-gencode=arch=compute_{major}{minor_str},code=sm_{major}{minor_str}"
    return load_inline(
        name="test_sr_fp16_subnormal_hw",
        cpp_sources=[_CPP_SOURCE],
        cuda_sources=[_CUDA_SOURCE],
        extra_include_paths=[_FLASHINFER_INCLUDE],
        extra_cuda_cflags=[gencode],
        functions=["sr_all_rand"],
        verbose=False,
    )


def _as_float(bits16: torch.Tensor) -> torch.Tensor:
    """Reinterpret raw fp16 bit patterns (int16) as float64 values."""
    return bits16.view(torch.float16).to(torch.float64)


def _fp16_ulp(x: torch.Tensor) -> torch.Tensor:
    """Spacing of fp16 values at |x| (fixed 2^-24 in the subnormal range)."""
    ax = x.abs().to(torch.float64)
    e = torch.floor(torch.log2(ax.clamp(min=_FP16_MIN_NORMAL)))
    return torch.where(
        ax < _FP16_MIN_NORMAL, torch.full_like(ax, _FP16_MIN_SUBNORMAL), 2.0 ** (e - 10)
    )


def _magnitude_grid(lo: float, hi: float, n: int, seed: int) -> torch.Tensor:
    """Return n fp32 values log-spaced over [lo, hi] with random signs and jitter (deterministic per seed)."""
    g = torch.Generator().manual_seed(seed)
    mags = torch.logspace(
        torch.log10(torch.tensor(lo)),
        torch.log10(torch.tensor(hi)),
        n,
        dtype=torch.float64,
    )
    # jitter so values are not aligned to fp16 grid points
    mags = mags * (1.0 + 0.01 * torch.rand(n, generator=g, dtype=torch.float64))
    signs = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0).to(torch.float64)
    return (mags * signs).to(torch.float32)


# ---------------------------------------------------------------------------
# Test 1: unbiasedness over magnitudes 1e-8 ... 1e3 (any GPU)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 7])
def test_sw_sr_unbiased_full_range(sr_module, seed):
    """Mean of the software SR over all 8192 random draws equals x (within 1/8192 ulp), 1e-8..1e3."""
    x = _magnitude_grid(1e-8, 1e3, 256, seed).cuda()
    out = _as_float(sr_module.sr_all_rand(x, 0))  # (n, 8192)
    mean = out.mean(dim=1)
    xd = x.to(torch.float64)
    # Exact expectation is x up to the 1/8192 quantization of the round-up probability.
    tol = _fp16_ulp(xd) / _N_RAND * 1.001
    err = (mean - xd).abs()
    bad = torch.where(err > tol)[0]
    for i in bad[:10].tolist():
        print(
            f"  x={xd[i].item():.6e}: E[out]={mean[i].item():.6e}, |err|={err[i].item():.3e}, tol={tol[i].item():.3e}"
        )
    assert bad.numel() == 0, (
        f"{bad.numel()}/{x.numel()} inputs are biased beyond 1/8192 ulp"
    )


# ---------------------------------------------------------------------------
# Test 2: no flush to zero for fp16-subnormal inputs (any GPU)
# ---------------------------------------------------------------------------
def test_sw_sr_no_flush_to_zero(sr_module):
    """Subnormal-range inputs are not flushed to zero: P(out == 0) is exactly 1 - |x| / 2^-24."""
    # fp16-subnormal range [2^-24, 2^-14) plus values below the smallest subnormal.
    x = torch.cat(
        [
            _magnitude_grid(_FP16_MIN_SUBNORMAL, _FP16_MIN_NORMAL * 0.999, 128, 1),
            torch.tensor([2.0**-25, 3e-8, 1e-7, 1e-6, 6.0e-5], dtype=torch.float32),
        ]
    ).cuda()
    out = _as_float(sr_module.sr_all_rand(x, 0))
    p_zero = (out == 0).to(torch.float64).mean(dim=1)
    ax = x.abs().to(torch.float64)
    # Exact P(result == 0) = max(0, 1 - |x| / 2^-24) for |x| < 2^-24, else 0.
    expected = (1.0 - ax / _FP16_MIN_SUBNORMAL).clamp(min=0.0)
    assert torch.all((p_zero - expected).abs() <= 1.0 / _N_RAND + 1e-12), (
        f"max |P(0) - expected| = {(p_zero - expected).abs().max().item():.3e}"
    )
    # No subnormal-range input collapses to zero for every random draw.
    assert torch.all(p_zero[ax >= _FP16_MIN_SUBNORMAL] < 1.0)
    # Signs are preserved for nonzero results.
    nz = out != 0
    assert torch.all(
        torch.sign(out[nz])
        == torch.sign(x.to(torch.float64)).unsqueeze(1).expand_as(out)[nz]
    )


# ---------------------------------------------------------------------------
# Test 3: bit-identity with the previous formulation for |x| >= 2^-14 (any GPU)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 42])
def test_sw_sr_bit_identical_normal_range(sr_module, seed):
    """For |x| >= 2^-14 (and overflow edges / NaN) the result is bit-identical to the previous formulation."""
    edge = torch.tensor(
        [
            _FP16_MIN_NORMAL,
            -_FP16_MIN_NORMAL,
            1.0,
            65504.0,
            65519.0,
            65520.0,
            65535.99,
            -65520.0,
            65536.0,
            1e6,
            float("nan"),
        ],
        dtype=torch.float32,
    )
    x = torch.cat([_magnitude_grid(_FP16_MIN_NORMAL, 65000.0, 512, seed), edge]).cuda()
    new = sr_module.sr_all_rand(x, 0)
    ref = sr_module.sr_all_rand(x, 1)
    mismatches = (new != ref).sum().item()
    if mismatches:
        i, r = torch.nonzero(new != ref)[0].tolist()
        print(
            f"  x={x[i].item():.8e}, rand13={r}: new=0x{new[i, r].item() & 0xFFFF:04X}, "
            f"reference=0x{ref[i, r].item() & 0xFFFF:04X}"
        )
    assert mismatches == 0, f"{mismatches}/{new.numel()} (x, rand13) pairs differ"


# ---------------------------------------------------------------------------
# Test 3b: Inf is preserved (any GPU)
# ---------------------------------------------------------------------------
def test_sw_sr_inf_preserved(sr_module):
    """+/-Inf stays +/-Inf for every random draw (the previous formulation produced NaN)."""
    x = torch.tensor([float("inf"), float("-inf")], dtype=torch.float32).cuda()
    bits = sr_module.sr_all_rand(x, 0).to(torch.int32) & 0xFFFF
    assert torch.all(bits[0] == 0x7C00), "+Inf must stay +Inf for every random draw"
    assert torch.all(bits[1] == 0xFC00), "-Inf must stay -Inf for every random draw"


# ---------------------------------------------------------------------------
# Test 4: hardware cvt.rs vs software (cvt.rs GPUs only)
# ---------------------------------------------------------------------------
def test_sw_sr_matches_hw_distribution(sr_module, sr_hw_module):
    """On cvt.rs GPUs: bitwise equal to hardware for fp16-normal inputs, same mean for subnormal inputs."""
    normal = _magnitude_grid(_FP16_MIN_NORMAL, 65000.0, 256, 3).cuda()
    hw = sr_hw_module.sr_all_rand(normal, 2)
    sw = sr_module.sr_all_rand(normal, 0)
    assert (hw != sw).sum().item() == 0, (
        "fp16-normal results must be bit-identical (sw vs hw)"
    )

    sub = _magnitude_grid(1e-8, _FP16_MIN_NORMAL * 0.999, 256, 4).cuda()
    hw_mean = _as_float(sr_hw_module.sr_all_rand(sub, 2)).mean(dim=1)
    sw_mean = _as_float(sr_module.sr_all_rand(sub, 0)).mean(dim=1)
    xd = sub.to(torch.float64)
    tol = _FP16_MIN_SUBNORMAL / _N_RAND * 2.001
    assert torch.all((hw_mean - xd).abs() <= tol), (
        "hardware cvt.rs is expected to be unbiased for subnormals"
    )
    assert torch.all((sw_mean - hw_mean).abs() <= tol), (
        f"max |E_sw - E_hw| = {(sw_mean - hw_mean).abs().max().item():.3e}"
    )
