/*
 * Copyright (c) 2026 by FlashInfer team.
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

#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nvshmem.h>

#include <cuda/ptx>

#include "tvm/ffi/container/map.h"

namespace flashinfer {
namespace mixed_comm {

using tvm::ffi::Map;

using T_ACC = float;

constexpr int WARP_SIZE = 32;
constexpr int ACCESS_BYTES = 16;
constexpr int NUM_BUFFERS = 4;
constexpr uint32_t NEG_ZERO_UINT32 = 0x80008000;

enum MixedCommOp {
  ALLREDUCE,
  ALLGATHER,
  REDUCESCATTER,
  ALLREDUCE_ALLGATHER,
  REDUCESCATTER_ALLREDUCE,
};

enum MixedCommMode {
  OPT_WAITS_MC,
  OPT_WAITS_UC,
  OPT_BYTES1_MC,
  OPT_BYTES1_UC,
  OPT_BYTES2_MC,
  OPT_BYTES2_UC,
};

__host__ __device__ constexpr bool is_opt_waits_mode(MixedCommMode mode) {
  return mode == MixedCommMode::OPT_WAITS_MC || mode == MixedCommMode::OPT_WAITS_UC;
}

__host__ __device__ constexpr bool is_opt_bytes1_mode(MixedCommMode mode) {
  return mode == MixedCommMode::OPT_BYTES1_MC || mode == MixedCommMode::OPT_BYTES1_UC;
}

__host__ __device__ constexpr bool is_opt_bytes2_mode(MixedCommMode mode) {
  return mode == MixedCommMode::OPT_BYTES2_MC || mode == MixedCommMode::OPT_BYTES2_UC;
}

__host__ __device__ constexpr bool is_mc_mode(MixedCommMode mode) {
  return mode == MixedCommMode::OPT_WAITS_MC || mode == MixedCommMode::OPT_BYTES1_MC ||
         mode == MixedCommMode::OPT_BYTES2_MC;
}

__host__ __device__ constexpr bool is_uc_mode(MixedCommMode mode) {
  return mode == MixedCommMode::OPT_WAITS_UC || mode == MixedCommMode::OPT_BYTES1_UC ||
         mode == MixedCommMode::OPT_BYTES2_UC;
}

template <bool use_local_tp, bool use_local_dp, bool use_inter_tp, bool use_inter_dp>
__host__ __device__ constexpr bool is_valid_op(MixedCommOp op) {
  constexpr bool use_tp = use_local_tp || use_inter_tp;
  constexpr bool use_dp = use_local_dp || use_inter_dp;
  constexpr bool use_mixed = use_tp && use_dp;
  if constexpr (use_mixed) {
    return op == MixedCommOp::ALLREDUCE_ALLGATHER || op == MixedCommOp::REDUCESCATTER_ALLREDUCE;
  } else if constexpr (use_tp) {
    return op == MixedCommOp::ALLREDUCE;
  } else {
    static_assert(use_dp, "use_dp must be true");
    return op == MixedCommOp::ALLGATHER || op == MixedCommOp::REDUCESCATTER;
  }
}

template <bool use_local_tp, bool use_inter_tp>
__host__ __device__ constexpr bool is_valid_mode(MixedCommMode mode) {
  if (is_opt_waits_mode(mode)) {
    return true;
  }
  if (is_opt_bytes1_mode(mode)) {
    return use_local_tp;
  }
  if (is_opt_bytes2_mode(mode)) {
    return use_inter_tp;
  }
  return false;
}

__host__ __device__ constexpr bool is_valid_block_y(int block_size_y, int local_tp_size,
                                                    int local_dp_size, MixedCommMode mode) {
  if (!is_mc_mode(mode)) {
    return block_size_y == 1;
  } else if (is_opt_waits_mode(mode)) {
    return local_dp_size % block_size_y == 0;
  } else {
    return (local_tp_size * local_dp_size) % block_size_y == 0;
  }
}

__host__ __device__ constexpr bool is_power_of_2(int x) { return (x > 0) && ((x & (x - 1)) == 0); }

template <int y, typename T>
__host__ __device__ constexpr T mod(T x) {
  if constexpr (is_power_of_2(y)) {
    return x & (y - 1);
  } else {
    return x % y;
  }
}

template <int y, typename T>
__host__ __device__ constexpr T floor_div(T x) {
  if constexpr (is_power_of_2(y)) {
    constexpr int shift = []() constexpr {
      int out = 0;
      int val = y;
      while (val > 1) {
        val >>= 1;
        ++out;
      }
      return out;
    }();
    return x >> shift;
  } else {
    return x / y;
  }
}

template <typename T_x, typename T_y>
__host__ __device__ constexpr T_x ceil_div(T_x x, T_y y) {
  return (x + (y - 1)) / y;
}

template <int y, typename T>
__host__ __device__ constexpr T ceil_div(T x) {
  return floor_div<y>(x + (y - 1));
}

template <typename T_x, typename T_y>
__host__ __device__ constexpr T_x round_down(T_x x, T_y y) {
  return (x / y) * y;
}

template <int y, typename T>
__host__ __device__ constexpr T round_down(T x) {
  if constexpr (is_power_of_2(y)) {
    return x & ~(y - 1);
  } else {
    return round_down(x, y);
  }
}

template <typename T_x, typename T_y>
__host__ __device__ constexpr T_x round_up(T_x x, T_y y) {
  return round_down(x + (y - 1), y);
}

template <int y, typename T>
__host__ __device__ constexpr T round_up(T x) {
  return round_down<y>(x + (y - 1));
}

template <typename T>
__device__ __forceinline__ bool has_neg_zero(T const* data) {
  static_assert(std::is_same_v<T, nv_half> || std::is_same_v<T, nv_bfloat16>, "Invalid data type");
  uint4 const* val = reinterpret_cast<uint4 const*>(data);
  return val->x == NEG_ZERO_UINT32 || val->y == NEG_ZERO_UINT32 || val->z == NEG_ZERO_UINT32 ||
         val->w == NEG_ZERO_UINT32;
}

template <typename T>
__device__ __forceinline__ void replace_neg_zero(T* data) {
  static_assert(std::is_same_v<T, nv_half> || std::is_same_v<T, nv_bfloat16>, "Invalid data type");
  uint4* val = reinterpret_cast<uint4*>(data);
  val->x = (val->x == NEG_ZERO_UINT32) ? 0 : val->x;
  val->y = (val->y == NEG_ZERO_UINT32) ? 0 : val->y;
  val->z = (val->z == NEG_ZERO_UINT32) ? 0 : val->z;
  val->w = (val->w == NEG_ZERO_UINT32) ? 0 : val->w;
}

template <typename T>
__device__ __forceinline__ void fill_neg_zero(T* data) {
  static_assert(std::is_same_v<T, nv_half> || std::is_same_v<T, nv_bfloat16>, "Invalid data type");
  *reinterpret_cast<uint4*>(data) =
      make_uint4(NEG_ZERO_UINT32, NEG_ZERO_UINT32, NEG_ZERO_UINT32, NEG_ZERO_UINT32);
}

template <bool use_inter, typename T>
struct BufferInfo {
  __device__ __forceinline__ BufferInfo(uint2* gmem_, int64_t vm_buffer_size_all_,
                                        int64_t ns_data_size_all_, int ns_signal_size_all_)
      : gmem{gmem_},
        vm_buffer_size_all{vm_buffer_size_all_},
        ns_data_size_all{ns_data_size_all_},
        ns_signal_size_all{ns_signal_size_all_} {
    *reinterpret_cast<uint2*>(info) = *gmem;
    update_buffer_index();
  }

  __device__ __forceinline__ void write_gmem(int tid) {
    if (tid == 0) {
      *gmem = *reinterpret_cast<uint2*>(info);
    }
  }

  __device__ __forceinline__ uint8_t& vm_buffer_index() {
    return reinterpret_cast<uint8_t*>(&info[0])[0];
  }

  __device__ __forceinline__ uint8_t& ns_buffer_index() {
    return reinterpret_cast<uint8_t*>(&info[0])[1];
  }

  __device__ __forceinline__ uint8_t& ns_reset_size() {
    return reinterpret_cast<uint8_t*>(&info[0])[2];
  }

  __device__ __forceinline__ int& vm_reset_size() { return info[1]; }

  __device__ __forceinline__ int64_t vm_ofst_buffer() {
    return vm_buffer_index() * vm_buffer_size_all;
  }

  __device__ __forceinline__ int64_t vm_ofst_buffer_prev() {
    uint8_t const& buffer_index = vm_buffer_index();
    return ((buffer_index == 0) ? (NUM_BUFFERS - 1) : (buffer_index - 1)) * vm_buffer_size_all;
  }

  __device__ __forceinline__ int64_t ns_ofst_data() { return ns_buffer_index() * ns_data_size_all; }

  __device__ __forceinline__ int64_t ns_ofst_data_prev() {
    uint8_t const& buffer_index = ns_buffer_index();
    return ((buffer_index == 0) ? (NUM_BUFFERS - 1) : (buffer_index - 1)) * ns_data_size_all;
  }

  __device__ __forceinline__ int ns_ofst_signal() { return ns_buffer_index() * ns_signal_size_all; }

  __device__ __forceinline__ int ns_ofst_signal_prev() {
    uint8_t const& buffer_index = ns_buffer_index();
    return ((buffer_index == 0) ? (NUM_BUFFERS - 1) : (buffer_index - 1)) * ns_signal_size_all;
  }

  __device__ __forceinline__ void update_buffer_index() {
    vm_buffer_index() = (vm_buffer_index() == NUM_BUFFERS - 1) ? 0 : (vm_buffer_index() + 1);
    if constexpr (use_inter) {
      ns_buffer_index() = (ns_buffer_index() == NUM_BUFFERS - 1) ? 0 : (ns_buffer_index() + 1);
    }
  }

  __device__ __forceinline__ void update_vm_reset_size(int reset_size) {
    vm_reset_size() = reset_size;
  }

  __device__ __forceinline__ void update_ns_reset_size(int reset_size) {
    ns_reset_size() = reset_size;
  }

  __device__ __forceinline__ void reset_vm_buffer(T* ptr, int idx, int stride) {
    __syncthreads();
    for (int i = idx; i < vm_reset_size(); i += stride) {
      fill_neg_zero(ptr + i);
    }
  }

  __device__ __forceinline__ void reset_ns_signal(uint64_t* ptr) {
    if (ns_reset_size() == 2) {
      *reinterpret_cast<uint4*>(ptr) = make_uint4(0, 0, 0, 0);
    } else {
      *ptr = 0;
    }
  }

  uint2* gmem;
  int info[2];
  int64_t vm_buffer_size_all;
  int64_t ns_data_size_all;
  int ns_signal_size_all;
};

template <int local_tp_size, int local_dp_size, bool use_inter_tp, bool use_inter_dp,
          MixedCommMode mode, typename T>
struct MixedCommArgs {
  static constexpr int local_size = local_tp_size * local_dp_size;
  static constexpr bool use_local_tp = local_tp_size > 1;
  static constexpr bool use_local_dp = local_dp_size > 1;
  static constexpr bool use_inter = use_inter_tp || use_inter_dp;
  static constexpr bool use_multicast = is_mc_mode(mode);
  static constexpr int elems_per_thd = ACCESS_BYTES / sizeof(T);
  static_assert(local_size > 1, "local_size must be greater than 1");

  MixedCommArgs(void* x_out_all_, void const* x_in_all_, void* uc_buffer_array_all_,
                void* mem_buffer_all_, void* mc_buffer_full_all_, void* mc_buffer_tp_all_,
                void* ns_buffer_all_, int vm_buffer_bytes_, int64_t ns_data_bytes_all_,
                int ns_signal_bytes_all_, int grid_size_, Map<int, int> max_block_size_dict_,
                int min_block_size_, int min_num_steps_, int64_t num_elems_all_, int local_tp_rank_,
                int local_dp_rank_, int inter_tp_rank_, int inter_dp_rank_, int inter_tp_size_,
                int inter_dp_size_)
      : x_out_all{reinterpret_cast<T*>(x_out_all_)},
        x_in_all{reinterpret_cast<T const*>(x_in_all_)},
        uc_buffer_array_all{reinterpret_cast<T* const*>(uc_buffer_array_all_)},
        mem_buffer_all{reinterpret_cast<T*>(mem_buffer_all_)},
        mc_buffer_full_all{reinterpret_cast<T*>(mc_buffer_full_all_)},
        mc_buffer_tp_all{reinterpret_cast<T*>(mc_buffer_tp_all_)},
        num_elems_all{num_elems_all_},
        grid_size{grid_size_},
        vm_buffer_size{floor_div<sizeof(T)>(vm_buffer_bytes_)},
        vm_buffer_size_all{static_cast<int64_t>(grid_size) * vm_buffer_size},
        ns_data_size_all{floor_div<sizeof(T)>(ns_data_bytes_all_)},
        ns_signal_size_all{floor_div<sizeof(uint64_t)>(ns_signal_bytes_all_)},
        local_tp_rank{local_tp_rank_},
        local_dp_rank{local_dp_rank_},
        inter_tp_rank{inter_tp_rank_},
        inter_dp_rank{inter_dp_rank_},
        inter_tp_size{inter_tp_size_},
        inter_dp_size{inter_dp_size_},
        local_rank{local_tp_rank + local_dp_rank * local_tp_size},
        inter_rank{inter_tp_rank + inter_dp_rank * inter_tp_size},
        inter_size{inter_tp_size * inter_dp_size},
        world_rank{local_rank + inter_rank * local_size},
        world_size{inter_size * local_size},
        tp_rank{local_tp_rank + inter_tp_rank * local_tp_size},
        tp_size{inter_tp_size * local_tp_size} {
    if constexpr (use_inter) {
      ns_data_all = reinterpret_cast<T*>(ns_buffer_all_);
      ns_signal_all = reinterpret_cast<uint64_t*>(ns_data_all + NUM_BUFFERS * ns_data_size_all);
    }
    buffer_info = reinterpret_cast<uint2*>(mem_buffer_all + NUM_BUFFERS * vm_buffer_size_all);
    int max_block_size = max_block_size_dict_[1];
    int min_block_size = std::min(min_block_size_, max_block_size);
    int64_t num_accesses_all = floor_div<elems_per_thd>(num_elems_all);
    int coef_tp;
    if constexpr (is_opt_waits_mode(mode)) {
      coef_tp = 1;
    } else if constexpr (is_opt_bytes1_mode(mode)) {
      coef_tp = local_tp_size;
    } else {
      static_assert(is_opt_bytes2_mode(mode), "Invalid mode");
      coef_tp = inter_tp_size * local_tp_size;
    }
    int64_t num_accesses_base_all = ceil_div(num_accesses_all, coef_tp);
    int64_t max_num_accesses_base = ceil_div(num_accesses_base_all, grid_size);
    if constexpr (use_inter) {
      if (max_num_accesses_base < min_block_size) {
        int active_grid_size = std::max<int>(num_accesses_base_all / min_block_size, 1);
        max_num_accesses_base = ceil_div(num_accesses_base_all, active_grid_size);
      }
    }
    max_num_elems = max_num_accesses_base * coef_tp * elems_per_thd;
    int64_t num_steps = ceil_div(max_num_accesses_base, min_block_size);
    if (num_steps > min_num_steps_) {
      num_steps = ceil_div(
          max_num_accesses_base,
          std::min<int64_t>(ceil_div(max_num_accesses_base, min_num_steps_), max_block_size));
    }
    block_size_x =
        round_up<WARP_SIZE>(std::max<int>(ceil_div(max_num_accesses_base, num_steps), inter_size));
    if constexpr (!use_multicast) {
      block_size_y = 1;
    } else if constexpr (is_opt_waits_mode(mode)) {
      block_size_y = local_dp_size;
    } else {
      block_size_y = local_size;
    }
    while (block_size_x * block_size_y > max_block_size_dict_[block_size_y]) {
      block_size_y >>= 1;
    }
    elems_per_blk = block_size_x * elems_per_thd;
  }

  T* x_out_all;
  T const* x_in_all;
  T* const* uc_buffer_array_all;
  T* mem_buffer_all;
  T* mc_buffer_full_all;
  T* mc_buffer_tp_all;
  T* ns_data_all;
  uint64_t* ns_signal_all;
  uint2* buffer_info;
  int64_t num_elems_all;
  int64_t max_num_elems;
  int grid_size;
  int block_size_x;
  int block_size_y;
  int vm_buffer_size;
  int64_t vm_buffer_size_all;
  int64_t ns_data_size_all;
  int ns_signal_size_all;
  int elems_per_blk;
  int local_tp_rank;
  int local_dp_rank;
  int inter_tp_rank;
  int inter_dp_rank;
  int inter_tp_size;
  int inter_dp_size;
  int local_rank;
  int inter_rank;
  int inter_size;
  int world_rank;
  int world_size;
  int tp_rank;
  int tp_size;
};

template <int block_size_y, int local_tp_size, int local_dp_size, bool use_inter_tp,
          bool use_inter_dp, MixedCommMode mode, typename T>
__global__ void allreduce_kernel(
    MixedCommArgs<local_tp_size, local_dp_size, use_inter_tp, use_inter_dp, mode, T> const args);

template <int block_size_y, int local_tp_size, int local_dp_size, bool use_inter_tp,
          bool use_inter_dp, MixedCommMode mode, typename T>
__global__ void allgather_kernel(
    MixedCommArgs<local_tp_size, local_dp_size, use_inter_tp, use_inter_dp, mode, T> const args);

template <int block_size_y, int local_tp_size, int local_dp_size, bool use_inter_tp,
          bool use_inter_dp, MixedCommMode mode, typename T>
__global__ void reducescatter_kernel(
    MixedCommArgs<local_tp_size, local_dp_size, use_inter_tp, use_inter_dp, mode, T> const args);

template <int block_size_y, int local_tp_size, int local_dp_size, bool use_inter_tp,
          bool use_inter_dp, MixedCommMode mode, typename T>
__global__ void fused_allreduce_allgather_kernel(
    MixedCommArgs<local_tp_size, local_dp_size, use_inter_tp, use_inter_dp, mode, T> const args);

template <int block_size_y, int local_tp_size, int local_dp_size, bool use_inter_tp,
          bool use_inter_dp, MixedCommMode mode, typename T>
__global__ void fused_reducescatter_allreduce_kernel(
    MixedCommArgs<local_tp_size, local_dp_size, use_inter_tp, use_inter_dp, mode, T> const args);

}  // namespace mixed_comm
}  // namespace flashinfer
