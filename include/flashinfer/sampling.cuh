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
#ifndef FLASHINFER_SAMPLING_CUH_
#define FLASHINFER_SAMPLING_CUH_

#include "../vec_dtypes.cuh"

namespace flashinfer {

namespace sampling {

/*!
 * \brief Top-P sampling kernel, pin probs in shared memory
 * \param x The input tensor of shape (N, D)
 * \param indices The output indices tensor of shape (N,)
 * \param p The probability threshold
 * \param d The 
 */
template <typename DType, typename IdType>
__global__ void TopPSamplingSMEMKernel(
  DType* __restrict__ x,
  IdType* __restrict__ indices,
  const float p,
  const uint32_t d
) {
  const uint32_t i = blockIdx.x;
  constexpr uint32_t vec_size = 16 / sizeof(DType);

}

/*!
 * \brief Top-P sampling kernel
 * \param x The input tensor of shape (N, D)
 * \param indices The output indices tensor of shape (N,)
 * \param p The probability threshold
 * \param d The 
 */
template <typename DType, typename IdType>
__global__ void TopPSamplingKernel(
  DType* __restrict__ x,
  IdType* __restrict__ indices,
  const float p,
  const uint32_t d
) {
  const uint32_t i = blockIdx.x;
  constexpr uint32_t vec_size = 16 / sizeof(DType);

}


} // namespace sampling

} // namespace flashinfer

#endif  // FLASHINFER_SAMPLING_CUH_
