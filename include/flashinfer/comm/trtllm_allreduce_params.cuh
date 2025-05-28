/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_TRTLLM_ALLREDUCE_PARAMS_CUH
#define FLASHINFER_TRTLLM_ALLREDUCE_PARAMS_CUH

#include <cstdint>

namespace flashinfer {

namespace trtllm_allreduce {

enum class AllReduceFusionOp : int8_t {
  NONE = 0,
  RESIDUAL_RMS_NORM = 1,
  LAST_PROCESS_FOR_UB = 2,
  RESIDUAL_RMS_PREPOST_NORM = 3,
  RESIDUAL_RMS_NORM_QUANT_FP8 = 4,
  RESIDUAL_RMS_NORM_QUANT_NVFP4 = 5,
  RESIDUAL_RMS_NORM_OUT_QUANT_FP8 = 6,
  RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4 = 7,
  MOE_ALLREDUCE_RESIDUAL_RMS_NORM = 8,
};

template <typename T>
struct AllReduceParams {
  size_t elts_total;
  size_t elts_per_rank;
  size_t elts_per_block;
  size_t rank_offset;
  size_t ranks_per_node;
  size_t local_rank;
  uint32_t barrier_flag;
  uint32_t* peer_barrier_ptrs_in[MAX_RANKS_PER_NODE];
  uint32_t* peer_barrier_ptrs_out[MAX_RANKS_PER_NODE];
  void* peer_comm_buffer_ptrs[MAX_RANKS_PER_NODE];
  void* local_output_buffer_ptr;
  void const* local_input_buffer_ptr;

  AllReduceFusionParams fusion_params;

  static AllReduceParams deserialize(int64_t* buffer, size_t tpSize, size_t tpRank, int token_num,
                                     int hidden_size, AllReduceFusionOp op) {
    void* const* buffer_ptrs = reinterpret_cast<void* const*>(buffer);
    int flag_offset;
    if (op == AllReduceFusionOp::RESIDUAL_RMS_NORM &&
        reduce_fusion::is_lamport_supported(dataType, token_num, hidden_size)) {
      flag_offset = 0;
    } else {
      flag_offset = 1;
    }
    auto const flag_ptr = &buffer[NUM_POINTERS_PER_RANK * tpSize + flag_offset];
    // cannot use 0 since 0 represents released state for barrier
    *flag_ptr += 1;
    uint32_t flag_value = *flag_ptr;
    AllReduceParams params;
    // Even plugins use ping buffers, odd plugins use pong.
    // That way, we don't need to wait for other GPUs to be done
    // before copying input tensor to workspace.
    auto const buffer_offset = (flag_value % 2 == 0) ? 0 : tpSize;

    for (int i = 0; i < tpSize; ++i) {
      params.peer_comm_buffer_ptrs[i] = buffer_ptrs[buffer_offset + i];
    }
    for (int i = 0; i < tpSize; ++i) {
      params.peer_barrier_ptrs_in[i] = reinterpret_cast<uint32_t*>(buffer_ptrs[2 * tpSize + i]);
    }
    for (int i = 0; i < tpSize; ++i) {
      params.peer_barrier_ptrs_out[i] = reinterpret_cast<uint32_t*>(buffer_ptrs[3 * tpSize + i]);
    }
    params.barrier_flag = flag_value;
    params.ranks_per_node = tpSize;
    params.local_rank = tpRank;

    return params;
  }
};

}  // namespace trtllm_allreduce

}  // namespace flashinfer

#endif  // FLASHINFER_TRTLLM_ALLREDUCE_PARAMS_CUH
