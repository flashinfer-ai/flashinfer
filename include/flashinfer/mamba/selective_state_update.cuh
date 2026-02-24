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

#include <cstdint>
#include <utility>

namespace flashinfer::mamba {

// Host-side algorithm selection for invokeSelectiveStateUpdate dispatch.
// Not stored in kernel params — no register overhead.
enum class SSUAlgorithm : int32_t {
  kAuto = 0,
  kSimple = 1,
  kVertical = 2,
  kHorizontal = 3,
};

inline const char* SSUAlgorithmToString(SSUAlgorithm algo) {
  switch (algo) {
    case SSUAlgorithm::kAuto:
      return "Auto";
    case SSUAlgorithm::kSimple:
      return "Simple";
    case SSUAlgorithm::kVertical:
      return "Vertical";
    case SSUAlgorithm::kHorizontal:
      return "Horizontal";
    default:
      return "Unknown";
  }
}

struct SelectiveStateUpdateParams {
  uint32_t batch{}, nheads{}, dim{}, dstate{}, ngroups{}, state_cache_size{};
  int32_t pad_slot_id{-1};

  int64_t x_stride_batch{}, dt_stride_batch{}, B_stride_batch{}, C_stride_batch{},
      out_stride_batch{}, z_stride_batch{}, state_stride_batch{};

  void* __restrict__ state{nullptr};
  void* __restrict__ x{nullptr};
  void* __restrict__ dt{nullptr};
  void* __restrict__ dt_bias{nullptr};
  void* __restrict__ A{nullptr};
  void* __restrict__ B{nullptr};
  void* __restrict__ C{nullptr};
  void* __restrict__ D{nullptr};
  void* __restrict__ z{nullptr};
  void* __restrict__ output{nullptr};
  void* __restrict__ state_batch_indices{nullptr};

  bool dt_softplus{false};
  bool update_state{true};
};

namespace mtp {
// Extended params struct for multi-token prediction (MTP)
struct SelectiveStateMTPParams : public SelectiveStateUpdateParams {
  uint32_t ntokens_mtp{1};
  uint64_t cache_steps{0};

  // MTP-specific strides for the token dimension
  int64_t x_stride_mtp{}, dt_stride_mtp{}, B_stride_mtp{}, C_stride_mtp{}, out_stride_mtp{},
      z_stride_mtp{};
  void* __restrict__ intermediate_states{
      nullptr};  // state_t: (ntokens_mtp, state_cache_size, nheads, dim, dstate)
  void* __restrict__ intermediate_state_indices{nullptr};  // (batch,)
  int64_t intermediate_state_stride_batch{};  // stride for batch dimension of intermediate_states
};
}  // namespace mtp

}  // namespace flashinfer::mamba

#include "kernel_selective_state_update_mtp.cuh"
#include "kernel_selective_state_update_stp.cuh"

#endif  // FLASHINFER_MAMBA_SELECTIVE_STATE_UPDATE_CUH_
