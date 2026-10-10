"""FP16 software stochastic rounding, including values missed by normal sampling."""

import pathlib

import pytest
import torch
from torch.utils.cpp_extension import load_inline


@pytest.fixture(scope="module")
def fp16_sr_sw():
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA")
    return load_inline(
        name="test_fp16_sr_sw_subnormals",
        cpp_sources="torch::Tensor convert(torch::Tensor x, torch::Tensor noise, int mode);",
        cuda_sources=r"""
#include <torch/extension.h>
#include <flashinfer/mamba/conversion.cuh>

__global__ void convert_kernel(const float* x, const int32_t* noise,
                               uint16_t* out, int n, int mode) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (mode == 2) {
    i *= 2;
    if (i + 1 < n) {
      uint32_t rbits = (noise[i] & 0x1FFFu) | ((noise[i + 1] & 0x1FFFu) << 16);
      uint32_t packed = flashinfer::mamba::conversion::cvt_rs_f16x2_f32(
          x[i], x[i + 1], rbits);
      out[i] = static_cast<uint16_t>(packed);
      out[i + 1] = static_cast<uint16_t>(packed >> 16);
    }
  } else if (i < n) {
    out[i] = mode == 0
        ? flashinfer::mamba::conversion::cvt_rs_f16_sw(x[i], noise[i])
        : __half_as_ushort(flashinfer::mamba::conversion::cvt_rs_f16_f32(x[i], noise[i]));
  }
}

torch::Tensor convert(torch::Tensor x, torch::Tensor noise, int mode) {
  auto out = torch::empty_like(x, x.options().dtype(torch::kFloat16));
  convert_kernel<<<(x.numel() + 255) / 256, 256>>>(
      x.data_ptr<float>(), noise.data_ptr<int32_t>(),
      reinterpret_cast<uint16_t*>(out.data_ptr<at::Half>()), x.numel(), mode);
  return out;
}
""",
        functions=["convert"],
        extra_include_paths=[
            str(pathlib.Path(__file__).resolve().parents[2] / "include")
        ],
        extra_cuda_cflags=["--expt-relaxed-constexpr"],
        verbose=False,
    )


@pytest.fixture(params=[0, 1, 2], ids=["software", "scalar", "packed"])
def conversion_mode(request):
    if request.param and torch.cuda.get_device_capability()[0] == 10:
        pytest.skip("Wrapper may use hardware cvt.rs on SM10x")
    return request.param


@pytest.mark.parametrize("noise", [0, 1, 4096, 8191])
def test_exact_finite_fp16_values(fp16_sr_sw, conversion_mode, noise):
    # Both signs of every finite FP16 value, including signed zeros. Conversion
    # to FP32 is exact, so stochastic rounding has no discarded bits to round.
    positive_bits = torch.arange(0x7C00, dtype=torch.int32)
    bits = torch.cat((positive_bits, positive_bits | 0x8000)).to(torch.int16)
    expected = bits.view(torch.float16).cuda()
    values = expected.float()
    random_bits = torch.full_like(values, noise, dtype=torch.int32)
    actual = fp16_sr_sw.convert(values, random_bits, conversion_mode)
    torch.testing.assert_close(
        actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
    )


def test_subnormal_rounding_boundaries(fp16_sr_sw, conversion_mode):
    # Independent reference: fixed subnormal ULPs, calculated in FP64 on CPU.
    # Sweep all 8192 noise values at underflow and subnormal/normal boundaries.
    fractions = torch.tensor(
        [
            0,
            2**-15,
            2**-13,
            0.25,
            0.5,
            0.75,
            1,
            1.25,
            1.5,
            1022.5,
            1023,
            1023.5,
            1024,
        ],
        dtype=torch.float32,
    )
    fractions = torch.cat(
        (
            fractions,
            torch.nextafter(fractions, torch.full_like(fractions, -float("inf"))),
        )
    )
    fractions = fractions[fractions >= 0]
    values = torch.cat((fractions, -fractions)) * 2**-24
    values = values.repeat_interleave(8192)
    random_bits = torch.arange(8192, dtype=torch.int32).repeat(len(values) // 8192)
    magnitude = torch.floor(values.double().abs() * 2**24 + random_bits.double() / 8192)
    sign = (values.view(torch.int32) < 0).to(torch.int32) * 0x8000
    expected_bits = (magnitude.to(torch.int32) | sign).to(torch.int16)
    # High random bits must not influence the 13-bit conversion.
    actual = fp16_sr_sw.convert(
        values.cuda(), (random_bits | 0xE000).cuda(), conversion_mode
    )
    torch.testing.assert_close(
        actual.view(torch.int16).cpu(), expected_bits, rtol=0, atol=0
    )
