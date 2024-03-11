/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_NORM_RMSNORM_CUH_
#define FLASHINFER_NORM_RMSNORM_CUH_

#include "../vec_dtypes.cuh"

namespace flashinfer {

namespace norm {

/*!
 * \brief Fused RMSNorm CUDA Kernel
 * \param x The input tensor of shape (N, D)
 * \param y The output tensor of shape (N, D)
 * \param d The dimension of the input tensor
 */
template <typename T>
__global__ void RMSNormKernel(
  T* __restrict__ x,
  T* __restrict__ y,
  const uint32_t d
) {
  const uint32_t i = blockIdx.x;
  constexpr uint32_t vec_size = 16 / sizeof(T);

}

}  // namespace norm

}

#endif  // FLASHINFER_NORM_RMSNORM_CUH_