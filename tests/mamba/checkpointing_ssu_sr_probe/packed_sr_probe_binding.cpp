// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#include <cstdint>
#include <vector>

// clang-format on
extern "C" cudaError_t launch_exact_packed_sr_probe(uint32_t const*, uint32_t const*,
                                                    uint32_t const*, uint32_t*, uint32_t*, int64_t,
                                                    cudaStream_t);

std::vector<at::Tensor> exact_packed_sr_probe(at::Tensor a, at::Tensor b, at::Tensor random) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && random.is_cuda(), "CUDA uint32 inputs required");
  TORCH_CHECK(a.scalar_type() == at::kUInt32 && b.scalar_type() == at::kUInt32 &&
                  random.scalar_type() == at::kUInt32,
              "raw uint32 bits required");
  TORCH_CHECK(a.dim() == 1 && b.sizes() == a.sizes() && random.sizes() == a.sizes(),
              "matching flat raw-bit arrays required");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && random.is_contiguous(),
              "contiguous raw-bit arrays required");
  TORCH_CHECK(a.device() == b.device() && a.device() == random.device(), "same device required");
  c10::cuda::CUDAGuard guard(a.device());
  auto reference = at::empty_like(a);
  auto candidate = at::empty_like(a);
  auto status = launch_exact_packed_sr_probe(
      a.const_data_ptr<uint32_t>(), b.const_data_ptr<uint32_t>(), random.const_data_ptr<uint32_t>(),
      reference.mutable_data_ptr<uint32_t>(), candidate.mutable_data_ptr<uint32_t>(), a.numel(),
      c10::cuda::getCurrentCUDAStream(a.get_device()).stream());
  TORCH_CHECK(status == cudaSuccess,
              "raw-bit CUDA probe launch failed: ", cudaGetErrorString(status));
  return {reference, candidate};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) { module.def("probe", &exact_packed_sr_probe); }
