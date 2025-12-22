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

namespace flashinfer::mamba {

struct SelectiveStateUpdateParams {
  uint32_t batch{}, nheads{}, dim{}, dstate{}, ngroups{}, state_cache_size{};
  int32_t pad_slot_id{-1};
  bool dt_softplus{false};

  int64_t x_stride_batch{}, dt_stride_batch{}, B_stride_batch{}, C_stride_batch{},
      out_stride_batch{};

  void* __restrict__ state{nullptr};  // state_t: (state_cache_size, nheads, dim, dstate)
  void* __restrict__ x{nullptr};      // input_t: (batch, nheads, dim)
  void* __restrict__ dt{nullptr};  // weight_t: (batch, nheads) but pretends to be (batch, nheads, dim)
  void* __restrict__ dt_bias{nullptr};  // weight_t (nheads) but pretends to be (nheads, dim)
  void* __restrict__ A{nullptr};  // matrixA_t: (nheads) but pretends to be (nheads, dim, dstate)
  void* __restrict__ B{nullptr};  // input_t: (batch, ngroups, dstate)
  void* __restrict__ C{nullptr};  // input_t: (batch, ngroups, dstate)
  void* __restrict__ D{nullptr};  // weight_t: (nheads) but pretends to be (nheads, dim)
  void* __restrict__ z{nullptr};  // input_t: (batch, nheads, dim)
  void* __restrict__ output{nullptr};               // input_t: (batch, nheads, dim)
  void* __restrict__ state_batch_indices{nullptr};  // state_batch_indices: (batch,)
};

template <typename input_t, typename weight_t, typename matrixA_t, typename state_t>
void invokeSelectiveStateUpdate( SelectiveStateUpdateParams& params, cudaStream_t stream)
{
  // This function is implemented in selective_state_update_kernel.cu
  throw std::runtime_error(
      "invokeSelectiveStateUpdate is not implemented for the given data types.");
}


}  // namespace flashinfer::mamba

#endif  // FLASHINFER_MAMBA_SELECTIVE_STATE_UPDATE_CUH_
