import pathlib

import pytest
import torch
from torch.utils.cpp_extension import load_inline


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_FLASHINFER_INCLUDE = str(_REPO_ROOT / "include")
_SPDLOG_INCLUDE = str(_REPO_ROOT / "3rdparty" / "spdlog" / "include")
_CUDA_FLAGS = [
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
]

_CPP_SOURCE = r"""
torch::Tensor test_block_reductions(int64_t num_threads);
"""

_CUDA_SOURCE = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <flashinfer/comm/trtllm_allreduce_fusion.cuh>

__global__ void block_reduction_kernel(float* output) {
  float sum = 1.0f;
  float max = static_cast<float>(threadIdx.x + 1);

  flashinfer::trtllm_allreduce_fusion::utils::blockReduceSumV2<float, 1>(&sum);
  flashinfer::trtllm_allreduce_fusion::utils::blockReduceMaxV2<float, 1>(&max);

  if (threadIdx.x == 0) {
    output[0] = sum;
    output[1] = max;
  }
}

torch::Tensor test_block_reductions(int64_t num_threads) {
  TORCH_CHECK(num_threads > 0 && num_threads <= 1024, "invalid block size");
  auto output = torch::empty({2}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  block_reduction_kernel<<<1, num_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
"""


@pytest.fixture(scope="module")
def reduction_module():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    gencode = f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"
    return load_inline(
        name="test_trtllm_allreduce_reduction",
        cpp_sources=[_CPP_SOURCE],
        cuda_sources=[_CUDA_SOURCE],
        extra_include_paths=[_FLASHINFER_INCLUDE, _SPDLOG_INCLUDE],
        extra_cuda_cflags=[*_CUDA_FLAGS, gencode],
        functions=["test_block_reductions"],
        verbose=False,
    )


@pytest.mark.parametrize("num_threads", [129, 159, 180, 192])
def test_block_reductions(reduction_module, num_threads):
    output = reduction_module.test_block_reductions(num_threads)
    expected = torch.tensor(
        [float(num_threads), float(num_threads)], device="cuda", dtype=torch.float32
    )
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
