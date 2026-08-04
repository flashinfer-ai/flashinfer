# SPDX-FileCopyrightText: (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pathlib

import pytest
import torch
from torch.utils.cpp_extension import load_inline


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_FLASHINFER_INCLUDE = str(_REPO_ROOT / "include")


_CPP_SOURCE = r"""
torch::Tensor test_uint_fastdiv();
"""

_CUDA_SOURCE = r"""
#include <torch/extension.h>

#include <flashinfer/fastdiv.cuh>

__global__ void uint_fastdiv_kernel(uint32_t* output) {
  const uint32_t n = 7;
  flashinfer::uint_fastdiv default_divisor;
  flashinfer::uint_fastdiv zero_divisor(0);
  flashinfer::uint_fastdiv three_divisor(3);

  if (threadIdx.x == 0) {
    uint32_t q, r;
    default_divisor.divmod(n, q, r);
    output[0] = q;
    output[1] = r;
    zero_divisor.divmod(n, q, r);
    output[2] = q;
    output[3] = r;
    three_divisor.divmod(n, q, r);
    output[4] = q;
    output[5] = r;
    output[6] = static_cast<unsigned int>(default_divisor);
    output[7] = static_cast<unsigned int>(zero_divisor);
    output[8] = static_cast<unsigned int>(three_divisor);
  }
}

torch::Tensor test_uint_fastdiv() {
  auto output = torch::empty({9}, torch::dtype(torch::kUInt32).device(torch::kCUDA));
  uint_fastdiv_kernel<<<1, 1>>>(output.data_ptr<uint32_t>());
  return output;
}
"""


@pytest.fixture(scope="module")
def fastdiv_module():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    gencode = f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"
    return load_inline(
        name="test_uint_fastdiv",
        cpp_sources=[_CPP_SOURCE],
        cuda_sources=[_CUDA_SOURCE],
        extra_include_paths=[_FLASHINFER_INCLUDE],
        extra_cuda_cflags=[gencode],
        functions=["test_uint_fastdiv"],
        verbose=False,
    )


def test_uint_fastdiv_zero_divisor_invariant(fastdiv_module):
    output = fastdiv_module.test_uint_fastdiv()
    expected = torch.tensor([7, 7, 7, 7, 2, 1, 0, 0, 3], device="cuda", dtype=torch.uint32)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
