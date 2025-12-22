/*
 * Copyright (c) 2025 by FlashInfer team.
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
#ifndef FLASHINFER_MAMBA_SELECTIVE_STATE_UPDATE_CUH_
#define FLASHINFER_MAMBA_SELECTIVE_STATE_UPDATE_CUH_

#include "../utils.cuh"
#include "../vec_dtypes.cuh"

namespace flashinfer {

namespace mamba {

// TODO: Add your kernel implementation here
// This should be framework-agnostic and use raw pointers only (no PyTorch/TensorFlow types)
//
// Example:
// template <typename T>
// __global__ void SelectiveStateUpdateKernel(
//     T* state, const T* x, const T* dt, const T* A, const T* B, const T* C,
//     int batch_size, int dim_state, int dim_input) {
//   // Your kernel implementation
// }

// TODO: Add host function that launches the kernel
// Example:
// template <typename T>
// cudaError_t SelectiveStateUpdate(
//     T* state, const T* x, const T* dt, const T* A, const T* B, const T* C,
//     int batch_size, int dim_state, int dim_input,
//     cudaStream_t stream = 0) {
//   // Compute grid/block dimensions
//   // Launch kernel
//   // Return cudaGetLastError()
// }

}  // namespace mamba

}  // namespace flashinfer

#endif  // FLASHINFER_MAMBA_SELECTIVE_STATE_UPDATE_CUH_