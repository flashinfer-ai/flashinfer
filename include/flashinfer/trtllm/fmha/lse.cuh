/*
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef FLASHINFER_TRTLLM_FMHA_LSE_CUH
#define FLASHINFER_TRTLLM_FMHA_LSE_CUH

#include <cuda.h>

#include <cmath>
#include <cub/device/device_transform.cuh>

#include "../../math.cuh"
#include "../../utils.cuh"

namespace flashinfer {

struct MDToLSE {
  __host__ __device__ float operator()(float2 md_elem) const {
    return math::log2e * md_elem.x + log2f(md_elem.y);
  }
};

inline cudaError_t ComputeLSEFromMD(float2* md, float* lse, int n, bool /*launch_with_pdl*/,
                                    cudaStream_t stream) {
  return cub::DeviceTransform::Transform(md, lse, n, MDToLSE{}, stream);
}

};  // namespace flashinfer

#endif  // FLASHINFER_TRTLLM_FMHA_LSE_CUH
