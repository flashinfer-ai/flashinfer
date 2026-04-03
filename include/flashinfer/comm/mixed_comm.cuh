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

namespace flashinfer {

namespace mixed_comm {

using T_ACC = float;

constexpr int WARP_SIZE = 32;
constexpr int ACCESS_BYTES = 16;

enum MixedCommMode {
  OPT_WAITS_SIGNAL_MC,
  OPT_WAITS_SIGNAL_UC,
  OPT_WAITS_LAMPORT1_MC,
  OPT_WAITS_LAMPORT1_UC,
  OPT_WAITS_LAMPORT2_MC,
  OPT_WAITS_LAMPORT2_UC,
  OPT_BYTES_SIGNAL_MC,
  OPT_BYTES_SIGNAL_UC,
  OPT_BYTES_LAMPORT1_MC,
  OPT_BYTES_LAMPORT1_UC,
  OPT_BYTES_LAMPORT2_MC,
  OPT_BYTES_LAMPORT2_UC,
  MIXED_SIGNAL_MC,
  MIXED_SIGNAL_UC,
  MIXED_LAMPORT1_MC,
  MIXED_LAMPORT1_UC,
  MIXED_LAMPORT2_MC,
  MIXED_LAMPORT2_UC,
};

constexpr int NUM_MIXED_COMM_MODES = MixedCommMode::MIXED_LAMPORT2_UC + 1;

union IndexInfo {
  uint32_t raw;
  uint8_t info[4];
};

template <typename T>
union ResetInfo {
  int4 raw;
  struct {
    T* data;
    size_t size;
  } info;
};

template <typename T>
struct MixedCommArgs {
  T* x_out;
  T const* x_in;
  T* const* mem_data_buffer;
  uint32_t* const* mem_signal_buffer;
  T* const* uc_data_array_buffer;
  T* const* mc_data_full_buffer;
  T* const* mc_data_tp_buffer;
  uint32_t* const* mc_signal_full_buffer;
  uint32_t* const* mc_signal_tp_buffer;
  T* const* ns_data_buffer;
  uint64_t* const* ns_signal_buffer;
  uint32_t* index_info;
  int4* reset_info;
  size_t num_elems_in;
  size_t num_elems_out;
  int local_tp_rank;
  int local_dp_rank;
  int inter_tp_rank;
  int inter_dp_rank;
  int inter_tp_size;
  int inter_dp_size;
};

__host__ __device__ constexpr bool is_opt_waits_mode(MixedCommMode mode) {
  return mode == MixedCommMode::OPT_WAITS_SIGNAL_MC || mode == MixedCommMode::OPT_WAITS_SIGNAL_UC ||
         mode == MixedCommMode::OPT_WAITS_LAMPORT1_MC ||
         mode == MixedCommMode::OPT_WAITS_LAMPORT1_UC ||
         mode == MixedCommMode::OPT_WAITS_LAMPORT2_MC ||
         mode == MixedCommMode::OPT_WAITS_LAMPORT2_UC;
}

__host__ __device__ constexpr bool is_opt_bytes_mode(MixedCommMode mode) {
  return mode == MixedCommMode::OPT_BYTES_SIGNAL_MC || mode == MixedCommMode::OPT_BYTES_SIGNAL_UC ||
         mode == MixedCommMode::OPT_BYTES_LAMPORT1_MC ||
         mode == MixedCommMode::OPT_BYTES_LAMPORT1_UC ||
         mode == MixedCommMode::OPT_BYTES_LAMPORT2_MC ||
         mode == MixedCommMode::OPT_BYTES_LAMPORT2_UC;
}

__host__ __device__ constexpr bool is_mixed_mode(MixedCommMode mode) {
  return mode == MixedCommMode::MIXED_SIGNAL_MC || mode == MixedCommMode::MIXED_SIGNAL_UC ||
         mode == MixedCommMode::MIXED_LAMPORT1_MC || mode == MixedCommMode::MIXED_LAMPORT1_UC ||
         mode == MixedCommMode::MIXED_LAMPORT2_MC || mode == MixedCommMode::MIXED_LAMPORT2_UC;
}

__host__ __device__ constexpr bool is_signal_mode(MixedCommMode mode) {
  return mode == MixedCommMode::OPT_WAITS_SIGNAL_MC || mode == MixedCommMode::OPT_WAITS_SIGNAL_UC ||
         mode == MixedCommMode::OPT_BYTES_SIGNAL_MC || mode == MixedCommMode::OPT_BYTES_SIGNAL_UC ||
         mode == MixedCommMode::MIXED_SIGNAL_MC || mode == MixedCommMode::MIXED_SIGNAL_UC;
}

__host__ __device__ constexpr bool is_lamport1_mode(MixedCommMode mode) {
  return mode == MixedCommMode::OPT_WAITS_LAMPORT1_MC ||
         mode == MixedCommMode::OPT_WAITS_LAMPORT1_UC ||
         mode == MixedCommMode::OPT_BYTES_LAMPORT1_MC ||
         mode == MixedCommMode::OPT_BYTES_LAMPORT1_UC || mode == MixedCommMode::MIXED_LAMPORT1_MC ||
         mode == MixedCommMode::MIXED_LAMPORT1_UC;
}

__host__ __device__ constexpr bool is_lamport2_mode(MixedCommMode mode) {
  return mode == MixedCommMode::OPT_WAITS_LAMPORT2_MC ||
         mode == MixedCommMode::OPT_WAITS_LAMPORT2_UC ||
         mode == MixedCommMode::OPT_BYTES_LAMPORT2_MC ||
         mode == MixedCommMode::OPT_BYTES_LAMPORT2_UC || mode == MixedCommMode::MIXED_LAMPORT2_MC ||
         mode == MixedCommMode::MIXED_LAMPORT2_UC;
}

__host__ __device__ constexpr bool is_mc_mode(MixedCommMode mode) {
  return mode == MixedCommMode::OPT_WAITS_SIGNAL_MC ||
         mode == MixedCommMode::OPT_WAITS_LAMPORT1_MC ||
         mode == MixedCommMode::OPT_WAITS_LAMPORT2_MC ||
         mode == MixedCommMode::OPT_BYTES_SIGNAL_MC ||
         mode == MixedCommMode::OPT_BYTES_LAMPORT1_MC ||
         mode == MixedCommMode::OPT_BYTES_LAMPORT2_MC || mode == MixedCommMode::MIXED_SIGNAL_MC ||
         mode == MixedCommMode::MIXED_LAMPORT1_MC || mode == MixedCommMode::MIXED_LAMPORT2_MC;
}

__host__ __device__ constexpr bool is_uc_mode(MixedCommMode mode) {
  return mode == MixedCommMode::OPT_WAITS_SIGNAL_UC ||
         mode == MixedCommMode::OPT_WAITS_LAMPORT1_UC ||
         mode == MixedCommMode::OPT_WAITS_LAMPORT2_UC ||
         mode == MixedCommMode::OPT_BYTES_SIGNAL_UC ||
         mode == MixedCommMode::OPT_BYTES_LAMPORT1_UC ||
         mode == MixedCommMode::OPT_BYTES_LAMPORT2_UC || mode == MixedCommMode::MIXED_SIGNAL_UC ||
         mode == MixedCommMode::MIXED_LAMPORT1_UC || mode == MixedCommMode::MIXED_LAMPORT2_UC;
}

__host__ __device__ constexpr bool single_node_vm_use_lamport(MixedCommMode mode) {
  return is_lamport1_mode(mode);
}

__host__ __device__ constexpr bool multi_node_vm_use_lamport(MixedCommMode mode) {
  return is_lamport2_mode(mode);
}

__host__ __device__ constexpr bool multi_node_ns_use_lamport(MixedCommMode mode) {
  return is_lamport1_mode(mode) || is_lamport2_mode(mode);
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
  if constexpr (y == (1 << 0)) {
    return x;
  } else if constexpr (y == (1 << 1)) {
    return x >> 1;
  } else if constexpr (y == (1 << 2)) {
    return x >> 2;
  } else if constexpr (y == (1 << 3)) {
    return x >> 3;
  } else if constexpr (y == (1 << 4)) {
    return x >> 4;
  } else if constexpr (y == (1 << 5)) {
    return x >> 5;
  } else if constexpr (y == (1 << 6)) {
    return x >> 6;
  } else if constexpr (y == (1 << 7)) {
    return x >> 7;
  } else if constexpr (y == (1 << 8)) {
    return x >> 8;
  } else if constexpr (y == (1 << 9)) {
    return x >> 9;
  } else if constexpr (y == (1 << 10)) {
    return x >> 10;
  } else if constexpr (y == (1 << 11)) {
    return x >> 11;
  } else if constexpr (y == (1 << 12)) {
    return x >> 12;
  } else if constexpr (y == (1 << 13)) {
    return x >> 13;
  } else if constexpr (y == (1 << 14)) {
    return x >> 14;
  } else if constexpr (y == (1 << 15)) {
    return x >> 15;
  } else if constexpr (y == (1 << 16)) {
    return x >> 16;
  } else if constexpr (y == (1 << 17)) {
    return x >> 17;
  } else if constexpr (y == (1 << 18)) {
    return x >> 18;
  } else if constexpr (y == (1 << 19)) {
    return x >> 19;
  } else if constexpr (y == (1 << 20)) {
    return x >> 20;
  } else if constexpr (y == (1 << 21)) {
    return x >> 21;
  } else if constexpr (y == (1 << 22)) {
    return x >> 22;
  } else if constexpr (y == (1 << 23)) {
    return x >> 23;
  } else if constexpr (y == (1 << 24)) {
    return x >> 24;
  } else if constexpr (y == (1 << 25)) {
    return x >> 25;
  } else if constexpr (y == (1 << 26)) {
    return x >> 26;
  } else if constexpr (y == (1 << 27)) {
    return x >> 27;
  } else if constexpr (y == (1 << 28)) {
    return x >> 28;
  } else if constexpr (y == (1 << 29)) {
    return x >> 29;
  } else if constexpr (y == (1 << 30)) {
    return x >> 30;
  } else if constexpr (y == (1 << 31)) {
    return x >> 31;
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
  static_assert(std::is_same_v<T, nv_half> || std::is_same_v<T, nv_bfloat16>,
                "Unsupported data type");
  constexpr uint32_t neg_zero = 0x80008000;
  uint32_t const* val = reinterpret_cast<uint32_t const*>(data);
  return val[0] == neg_zero || val[1] == neg_zero || val[2] == neg_zero || val[3] == neg_zero;
}

template <typename T>
__device__ __forceinline__ void replace_neg_zero(T* data) {
  static_assert(std::is_same_v<T, nv_half> || std::is_same_v<T, nv_bfloat16>,
                "Unsupported data type");
  constexpr uint32_t neg_zero = 0x80008000;
  uint32_t* val = reinterpret_cast<uint32_t*>(data);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    if (val[i] == neg_zero) {
      val[i] = 0;
    }
  }
}

template <typename T>
__device__ __forceinline__ void fill_neg_zero(T* data) {
  static_assert(std::is_same_v<T, nv_half> || std::is_same_v<T, nv_bfloat16>,
                "Unsupported data type");
  *reinterpret_cast<__uint128_t*>(data) = 0x80008000800080008000800080008000;
}

template <typename T, bool use_replace = false>
__device__ __forceinline__ void copy_data(T* outputs, T const* inputs) {
  *reinterpret_cast<int4*>(outputs) = *reinterpret_cast<int4 const*>(inputs);
  if constexpr (use_replace) {
    replace_neg_zero(outputs);
  }
}

template <typename T>
__device__ __forceinline__ void load_volatile(T* outputs, T const* inputs) {
  float* dst = reinterpret_cast<float*>(outputs);
  float const* src = reinterpret_cast<float const*>(inputs);
  asm volatile("ld.volatile.global.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3])
               : "l"(src)
               : "memory");
}

template <int comm_size, bool use_release = true>
__device__ __forceinline__ void multicast_sync(uint32_t* mc_signal, uint32_t* mem_signal) {
  if constexpr (use_release) {
    asm volatile("multimem.red.release.sys.global.add.u32 [%0], %1;"
                 :
                 : "l"(mc_signal), "n"(1)
                 : "memory");
  } else {
    asm volatile("multimem.red.relaxed.sys.global.add.u32 [%0], %1;"
                 :
                 : "l"(mc_signal), "n"(1)
                 : "memory");
  }
  cuda::atomic_ref<uint32_t, cuda::thread_scope_system> ref_signal(*mem_signal);
  while (ref_signal.load(cuda::memory_order_acquire) != comm_size);
}

template <typename T>
__device__ __forceinline__ void multicast_load_reduce(T* outputs, T const* inputs) {
  uint32_t* dst = reinterpret_cast<uint32_t*>(outputs);
  uint32_t const* src = reinterpret_cast<uint32_t const*>(inputs);
  if constexpr (std::is_same_v<T, nv_half>) {
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.f16x2 {%0, %1, %2, %3}, [%4];"
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
        : "l"(src)
        : "memory");
  } else {
    static_assert(std::is_same_v<T, nv_bfloat16>, "Unsupported data type");
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2 {%0, %1, %2, %3}, [%4];"
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
        : "l"(src)
        : "memory");
  }
}

template <typename T>
__device__ __forceinline__ void multicast_store(T* outputs, T const* inputs) {
  float* dst = reinterpret_cast<float*>(outputs);
  float const* src = reinterpret_cast<float const*>(inputs);
  asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(dst), "f"(src[0]), "f"(src[1]), "f"(src[2]), "f"(src[3])
               : "memory");
}

template <int comm_size, typename T>
__device__ __forceinline__ void unicast_store(T* const* outputs, T* inputs, size_t ofst) {
#pragma unroll
  for (int i = 0; i < comm_size; ++i) {
    copy_data(outputs[i] + ofst, inputs);
  }
}

template <int elems_per_thd, typename T_in, typename T_out>
__device__ __forceinline__ void convert_dtype(T_out* outputs, T_in const* inputs) {
  static_assert(!std::is_same_v<T_in, T_out>, "T_in and T_out must be different");
  static_assert(elems_per_thd % 2 == 0, "elems_per_thd must be even");
  if constexpr (std::is_same_v<T_in, float>) {
#pragma unroll
    for (int i = 0; i < elems_per_thd; i += 2) {
      if constexpr (std::is_same_v<T_out, nv_half>) {
        *reinterpret_cast<nv_half2*>(outputs + i) =
            __float22half2_rn(*reinterpret_cast<float2 const*>(inputs + i));
      } else {
        static_assert(std::is_same_v<T_out, nv_bfloat16>, "Unsupported data type");
        *reinterpret_cast<nv_bfloat162*>(outputs + i) =
            __float22bfloat162_rn(*reinterpret_cast<float2 const*>(inputs + i));
      }
    }
  } else {
    static_assert(std::is_same_v<T_out, float>, "Unsupported data type");
#pragma unroll
    for (int i = 0; i < elems_per_thd; i += 2) {
      if constexpr (std::is_same_v<T_in, nv_half>) {
        *reinterpret_cast<float2*>(outputs + i) =
            __half22float2(*reinterpret_cast<nv_half2 const*>(inputs + i));
      } else {
        static_assert(std::is_same_v<T_in, nv_bfloat16>, "Unsupported data type");
        *reinterpret_cast<float2*>(outputs + i) =
            __bfloat1622float2(*reinterpret_cast<nv_bfloat162 const*>(inputs + i));
      }
    }
  }
}

template <int elems_per_thd, typename T>
__device__ __forceinline__ void accumulate(T_ACC* outputs, T const* inputs) {
  T_ACC reg_cvt[elems_per_thd];
  convert_dtype<elems_per_thd>(reg_cvt, inputs);
#pragma unroll
  for (int i = 0; i < elems_per_thd; ++i) {
    outputs[i] += reg_cvt[i];
  }
}

template <bool use_lamport, typename T>
__device__ __forceinline__ void load_data(T* outputs, T const* inputs) {
  if constexpr (!use_lamport) {
    copy_data(outputs, inputs);
  } else {
    do {
      load_volatile(outputs, inputs);
    } while (has_neg_zero(outputs));
  }
}

template <int reduce_size, bool use_lamport, typename T, bool use_replace = false>
__device__ __forceinline__ void reduce_data(T* reg_data, T const* data, size_t stride) {
  constexpr int elems_per_thd = ACCESS_BYTES / sizeof(T);
  T_ACC reg_acc[elems_per_thd];
  if constexpr (!use_lamport) {
    copy_data(reg_data, data);
    convert_dtype<elems_per_thd>(reg_acc, reg_data);
#pragma unroll
    for (int i = 1; i < reduce_size; ++i) {
      data += stride;
      copy_data(reg_data, data);
      accumulate<elems_per_thd>(reg_acc, reg_data);
    }
    convert_dtype<elems_per_thd>(reg_data, reg_acc);
  } else {
    bool invalid;
    do {
      T const* data_val = data;
      load_volatile(reg_data, data_val);
      invalid = has_neg_zero(reg_data);
      convert_dtype<elems_per_thd>(reg_acc, reg_data);
#pragma unroll
      for (int i = 1; i < reduce_size; ++i) {
        data_val += stride;
        load_volatile(reg_data, data_val);
        if (has_neg_zero(reg_data)) {
          invalid = true;
        }
        accumulate<elems_per_thd>(reg_acc, reg_data);
      }
      convert_dtype<elems_per_thd>(reg_data, reg_acc);
    } while (invalid);
    if constexpr (use_replace) {
      replace_neg_zero(reg_data);
    }
  }
}

template <bool use_lamport, typename T, bool use_replace = false>
__device__ __forceinline__ void reduce_data(T* reg_data, T const* data, size_t stride,
                                            int reduce_size) {
  constexpr int elems_per_thd = ACCESS_BYTES / sizeof(T);
  T_ACC reg_acc[elems_per_thd];
  if constexpr (!use_lamport) {
    copy_data(reg_data, data);
    convert_dtype<elems_per_thd>(reg_acc, reg_data);
    for (int i = 1; i < reduce_size; ++i) {
      data += stride;
      copy_data(reg_data, data);
      accumulate<elems_per_thd>(reg_acc, reg_data);
    }
    convert_dtype<elems_per_thd>(reg_data, reg_acc);
  } else {
    bool invalid;
    do {
      T const* data_val = data;
      load_volatile(reg_data, data_val);
      invalid = has_neg_zero(reg_data);
      convert_dtype<elems_per_thd>(reg_acc, reg_data);
      for (int i = 1; i < reduce_size; ++i) {
        data_val += stride;
        load_volatile(reg_data, data_val);
        if (has_neg_zero(reg_data)) {
          invalid = true;
        }
        accumulate<elems_per_thd>(reg_acc, reg_data);
      }
      convert_dtype<elems_per_thd>(reg_data, reg_acc);
    } while (invalid);
    if constexpr (use_replace) {
      replace_neg_zero(reg_data);
    }
  }
}

template <int local_size, typename T>
__device__ __forceinline__ void set_uc_data_array(T* uc_data_array[local_size],
                                                  T* const* uc_data_all_array, int local_rank,
                                                  size_t ofst) {
  int idx = local_rank;
#pragma unroll
  for (int i = 0; i < local_size; ++i) {
    uc_data_array[i] = uc_data_all_array[idx] + ofst;
    idx = (idx == local_size - 1) ? 0 : (idx + 1);
  }
}

template <int local_tp_size, int local_dp_size, typename T>
__device__ __forceinline__ void set_uc_data_array(T* uc_data_array[local_tp_size * local_dp_size],
                                                  T* const* uc_data_all_array, int local_tp_rank,
                                                  int local_dp_rank, size_t ofst) {
  constexpr int local_size = local_tp_size * local_dp_size;
  int idx_base = local_dp_rank * local_tp_size;
#pragma unroll
  for (int i = 0; i < local_dp_size; ++i) {
    int idx_res = local_tp_rank;
#pragma unroll
    for (int j = 0; j < local_tp_size; ++j) {
      uc_data_array[i * local_tp_size + j] = uc_data_all_array[idx_base + idx_res] + ofst;
      idx_res = (idx_res == local_tp_size - 1) ? 0 : (idx_res + 1);
    }
    idx_base = (idx_base == local_size - local_tp_size) ? 0 : (idx_base + local_tp_size);
  }
}

template <bool vm_use_lamport>
__device__ __forceinline__ int get_vm_index(IndexInfo const& index_info) {
  int vm_index;
  if constexpr (!vm_use_lamport) {
    vm_index = index_info.info[0];
  } else {
    vm_index = index_info.info[1];
  }
  return vm_index;
}

template <bool ns_use_lamport>
__device__ __forceinline__ int get_ns_index(IndexInfo const& index_info) {
  int ns_index;
  if constexpr (!ns_use_lamport) {
    ns_index = index_info.info[2];
  } else {
    ns_index = index_info.info[3];
  }
  return ns_index;
}

template <bool vm_use_lamport>
__device__ __forceinline__ void update_index_info_vm(IndexInfo& index_info) {
  if constexpr (!vm_use_lamport) {
    index_info.info[0] = (index_info.info[0] == 2) ? 0 : (index_info.info[0] + 1);
  } else {
    index_info.info[1] = (index_info.info[1] == 5) ? 3 : (index_info.info[1] + 1);
  }
}

template <bool ns_use_lamport>
__device__ __forceinline__ void update_index_info_ns(IndexInfo& index_info) {
  if constexpr (!ns_use_lamport) {
    index_info.info[2] = (index_info.info[2] == 2) ? 0 : (index_info.info[2] + 1);
  } else {
    index_info.info[3] = (index_info.info[3] == 5) ? 3 : (index_info.info[3] + 1);
  }
}

template <int local_tp_size, int local_dp_size, int mode_val, typename T>
__global__ void fused_allreduce_allgather_single_node(MixedCommArgs<T> args) {
  constexpr MixedCommMode mode = static_cast<MixedCommMode>(mode_val);
  constexpr int local_size = local_tp_size * local_dp_size;
  int local_rank = args.local_tp_rank + args.local_dp_rank * local_tp_size;
  constexpr int elems_per_thd = ACCESS_BYTES / sizeof(T);
  int elems_per_blk = blockDim.x * elems_per_thd;
  int const& tid = threadIdx.x;
  int const& bid = blockIdx.x;
  int const ofst_index = tid * elems_per_thd;
  int const bid_2 = bid * 2;

  // Divide inputs and outputs into blocks
  size_t max_num_elems_in;
  if constexpr (is_opt_waits_mode(mode)) {
    max_num_elems_in = round_up<elems_per_thd>(ceil_div(args.num_elems_in, gridDim.x));
  } else {
    max_num_elems_in =
        round_up<elems_per_thd * local_tp_size>(ceil_div(args.num_elems_in, gridDim.x));
  }
  size_t ofst_elems_in = bid * max_num_elems_in;
  size_t num_elems_in = (args.num_elems_in > ofst_elems_in)
                            ? min(args.num_elems_in - ofst_elems_in, max_num_elems_in)
                            : 0;
  T* x_out = args.x_out + ofst_elems_in;
  T const* x_in = args.x_in + ofst_elems_in;

  // Prepare basic variables
  constexpr bool vm_use_lamport = single_node_vm_use_lamport(mode);
  constexpr bool use_multicast = is_mc_mode(mode);
  T reg_data[elems_per_thd];
  IndexInfo index_info;
  index_info.raw = args.index_info[bid];
  int vm_index = get_vm_index<vm_use_lamport>(index_info);

  if constexpr (is_opt_waits_mode(mode) || local_tp_size == 1) {
    // Max virtual memory buffer size:
    //   max_local_bytes * local_size
    size_t ofst_data = local_size * ofst_elems_in;
    size_t ofst_data_self = ofst_data + local_rank * max_num_elems_in;
    T *mc_data, *uc_data_array[local_size];
    if constexpr (use_multicast) {
      mc_data = args.mc_data_full_buffer[vm_index] + ofst_data_self;
    } else {
      set_uc_data_array<local_size>(uc_data_array,
                                    args.uc_data_array_buffer + vm_index * local_size, local_rank,
                                    ofst_data_self);
    }

    // Read inputs and gather data
    for (size_t i = ofst_index; i < num_elems_in; i += elems_per_blk) {
      copy_data<T, vm_use_lamport>(reg_data, x_in + i);
      if constexpr (use_multicast) {
        multicast_store(mc_data + i, reg_data);
      } else {
        unicast_store<local_size>(uc_data_array, reg_data, i);
      }
    }
    T* mem_data;
    if constexpr (use_multicast) {
      mem_data = args.mem_data_buffer[vm_index] + ofst_data;
    } else {
      mem_data = uc_data_array[0] - ofst_data_self + ofst_data;
    }
    size_t stride_mem_data = local_tp_size * max_num_elems_in;
    if constexpr (!vm_use_lamport) {
      __syncthreads();
      if (tid == 0) {
        update_index_info_vm<vm_use_lamport>(index_info);
        args.index_info[bid] = index_info.raw;
        uint32_t* mem_signal = args.mem_signal_buffer[vm_index] + bid;
        uint32_t* mc_signal = args.mc_signal_full_buffer[vm_index] + bid;
        multicast_sync<local_size>(mc_signal, mem_signal);
        *mem_signal = 0;
      }
      __syncthreads();
    } else {
      int4* reset_info = args.reset_info + bid_2;
      ResetInfo<T> vm_reset;
      vm_reset.raw = reset_info[0];
      for (size_t i = ofst_index; i < vm_reset.info.size; i += elems_per_blk) {
        fill_neg_zero(vm_reset.info.data + i);
      }
      __syncthreads();
      if (tid == 0) {
        update_index_info_vm<vm_use_lamport>(index_info);
        args.index_info[bid] = index_info.raw;
        vm_reset.info.data = mem_data;
        vm_reset.info.size = local_size * max_num_elems_in;
        reset_info[0] = vm_reset.raw;
      }
    }
    // Reduce data and write outputs
    for (size_t i = ofst_index; i < num_elems_in; i += elems_per_blk) {
      T* dst = x_out + i;
      T const* src = mem_data + i;
#pragma unroll
      for (int j = 0; j < local_dp_size; ++j) {
        if constexpr (local_tp_size == 1) {
          load_data<vm_use_lamport>(reg_data, src);
        } else {
          reduce_data<local_tp_size, vm_use_lamport>(reg_data, src, max_num_elems_in);
        }
        copy_data(dst, reg_data);
        dst += args.num_elems_in;
        src += stride_mem_data;
      }
    }
  } else {
    static_assert(is_opt_bytes_mode(mode) && local_tp_size > 1, "Unsupported mode");
    // Max virtual memory buffer size:
    //   max_local_bytes * min(local_size, local_dp_size + 1)
    size_t num_elems_in_chunk = floor_div<local_tp_size>(max_num_elems_in);
    size_t ofst_elems_in_chunk = args.local_tp_rank * num_elems_in_chunk;
    size_t ofst_elems_in_self = ofst_elems_in + ofst_elems_in_chunk;
    size_t ofst_data_out = max_num_elems_in * gridDim.x + local_dp_size * ofst_elems_in;
    size_t ofst_data_out_self =
        ofst_data_out + args.local_dp_rank * max_num_elems_in + ofst_elems_in_chunk;

    if constexpr (use_multicast && !vm_use_lamport) {
      // Read inputs
      T* mem_data_all = args.mem_data_buffer[vm_index];
      T* mem_data_in = mem_data_all + ofst_elems_in;
      for (size_t i = ofst_index; i < num_elems_in; i += elems_per_blk) {
        copy_data(mem_data_in + i, x_in + i);
      }
      T* mc_data_in = args.mc_data_tp_buffer[vm_index] + ofst_elems_in_self;
      T* mc_data_out;
      if constexpr (local_dp_size == 1) {
        mc_data_out = mc_data_in - ofst_elems_in_self + ofst_data_out_self;
      } else {
        mc_data_out = args.mc_data_full_buffer[vm_index] + ofst_data_out_self;
      }
      __syncthreads();
      uint32_t *mem_signal_in, *mc_signal_in;
      if (tid == 0) {
        mem_signal_in = args.mem_signal_buffer[vm_index] + bid_2;
        mc_signal_in = args.mc_signal_tp_buffer[vm_index] + bid_2;
        multicast_sync<local_tp_size, /*use_release=*/false>(mc_signal_in, mem_signal_in);
      }
      __syncthreads();
      // Reduce and gather data
      for (size_t i = ofst_index; i < num_elems_in_chunk; i += elems_per_blk) {
        multicast_load_reduce(reg_data, mc_data_in + i);
        multicast_store(mc_data_out + i, reg_data);
      }
      T* mem_data_out = mem_data_all + ofst_data_out;
      __syncthreads();
      if (tid == 0) {
        update_index_info_vm<vm_use_lamport>(index_info);
        args.index_info[bid] = index_info.raw;
        uint32_t* mem_signal_out = mem_signal_in + 1;
        uint32_t* mc_signal_out;
        if constexpr (local_dp_size == 1) {
          mc_signal_out = mc_signal_in + 1;
        } else {
          mc_signal_out = args.mc_signal_full_buffer[vm_index] + (bid_2 + 1);
        }
        multicast_sync<local_size>(mc_signal_out, mem_signal_out);
        *reinterpret_cast<uint64_t*>(mem_signal_in) = 0;
      }
      __syncthreads();
      // Write outputs
      for (size_t i = ofst_index; i < num_elems_in; i += elems_per_blk) {
        T* dst = x_out + i;
        T const* src = mem_data_out + i;
#pragma unroll
        for (int j = 0; j < local_dp_size; ++j) {
          copy_data(dst, src);
          dst += args.num_elems_in;
          src += max_num_elems_in;
        }
      }
    } else {
      // Read inputs and gather data
      T* uc_data_array[local_size];
      set_uc_data_array<local_tp_size, local_dp_size>(
          uc_data_array, args.uc_data_array_buffer + vm_index * local_size, args.local_tp_rank,
          args.local_dp_rank, ofst_elems_in_self);
      T* mem_data_all = uc_data_array[0] - ofst_elems_in_self;
      T* mem_data_in = mem_data_all + ofst_elems_in;
      size_t ofst_src_base = args.local_tp_rank * num_elems_in_chunk;
      for (size_t i = ofst_index; i < num_elems_in_chunk; i += elems_per_blk) {
#pragma unroll
        for (int j = 0; j < local_tp_size; ++j) {
          size_t ofst_src = ofst_src_base + i;
          if (ofst_src < num_elems_in) {
            copy_data<T, vm_use_lamport>(reg_data, x_in + ofst_src);
            copy_data(uc_data_array[j] + i, reg_data);
          }
          ofst_src_base += num_elems_in_chunk;
          if (ofst_src_base == max_num_elems_in) {
            ofst_src_base = 0;
          }
        }
      }
      if constexpr (vm_use_lamport) {
        size_t num_elems_src = (num_elems_in > ofst_src_base)
                                   ? min(num_elems_in - ofst_src_base, num_elems_in_chunk)
                                   : 0;
        for (size_t i = num_elems_src + ofst_index; i < num_elems_in_chunk; i += elems_per_blk) {
          T* mem_data_in_val = mem_data_in + i;
#pragma unroll
          for (int j = 0; j < local_tp_size; ++j) {
            *reinterpret_cast<__uint128_t*>(mem_data_in_val) = 0;
            mem_data_in_val += num_elems_in_chunk;
          }
        }
      }
      T* mc_data_out;
      if constexpr (use_multicast) {
        mc_data_out = args.mc_data_full_buffer[vm_index] + ofst_data_out_self;
      } else {
        size_t ofst_uc_data = ofst_data_out_self - ofst_elems_in_self;
#pragma unroll
        for (int i = 0; i < local_size; ++i) {
          uc_data_array[i] += ofst_uc_data;
        }
      }
      uint32_t *mem_signal_in, *mc_signal_in;
      int4* reset_info;
      ResetInfo<T> vm_reset;
      if constexpr (!vm_use_lamport) {
        __syncthreads();
        if (tid == 0) {
          mem_signal_in = args.mem_signal_buffer[vm_index] + bid_2;
          mc_signal_in = args.mc_signal_tp_buffer[vm_index] + bid_2;
          multicast_sync<local_tp_size>(mc_signal_in, mem_signal_in);
        }
        __syncthreads();
      } else {
        reset_info = args.reset_info + bid_2;
        vm_reset.raw = reset_info[0];
        for (size_t i = ofst_index; i < vm_reset.info.size; i += elems_per_blk) {
          fill_neg_zero(vm_reset.info.data + i);
        }
      }
      // Reduce and gather data
      for (size_t i = ofst_index; i < num_elems_in_chunk; i += elems_per_blk) {
        reduce_data<local_tp_size, vm_use_lamport, T, /*use_replace=*/true>(
            reg_data, mem_data_in + i, num_elems_in_chunk);
        if constexpr (use_multicast) {
          multicast_store(mc_data_out + i, reg_data);
        } else {
          unicast_store<local_size>(uc_data_array, reg_data, i);
        }
      }
      T* mem_data_out = mem_data_all + ofst_data_out;
      __syncthreads();
      if constexpr (!vm_use_lamport) {
        if (tid == 0) {
          update_index_info_vm<vm_use_lamport>(index_info);
          args.index_info[bid] = index_info.raw;
          uint32_t* mem_signal_out = mem_signal_in + 1;
          uint32_t* mc_signal_out;
          if constexpr (local_dp_size == 1) {
            mc_signal_out = mc_signal_in + 1;
          } else {
            mc_signal_out = args.mc_signal_full_buffer[vm_index] + (bid_2 + 1);
          }
          multicast_sync<local_size>(mc_signal_out, mem_signal_out);
          *reinterpret_cast<uint64_t*>(mem_signal_in) = 0;
        }
        __syncthreads();
      } else {
        for (size_t i = ofst_index; i < max_num_elems_in; i += elems_per_blk) {
          fill_neg_zero(mem_data_in + i);
        }
        if (tid == 0) {
          update_index_info_vm<vm_use_lamport>(index_info);
          args.index_info[bid] = index_info.raw;
          vm_reset.info.data = mem_data_out;
          vm_reset.info.size = local_dp_size * max_num_elems_in;
          reset_info[0] = vm_reset.raw;
        }
      }
      // Write outputs
      for (size_t i = ofst_index; i < num_elems_in; i += elems_per_blk) {
        T* dst = x_out + i;
        T const* src = mem_data_out + i;
#pragma unroll
        for (int j = 0; j < local_dp_size; ++j) {
          load_data<vm_use_lamport>(reg_data, src);
          copy_data(dst, reg_data);
          dst += args.num_elems_in;
          src += max_num_elems_in;
        }
      }
    }
  }
}

template <int local_tp_size, int local_dp_size, int mode, typename T>
__global__ void fused_allreduce_allgather_multi_node(MixedCommArgs<T> args) {
  // constexpr int local_size = local_tp_size * local_dp_size;
  // int local_rank = local_tp_rank + local_dp_rank * local_tp_size;
  // int num_nodes = inter_tp_size * inter_dp_size;
  // int node_id = inter_tp_rank + inter_dp_rank * inter_tp_size;
  // int dp_size = local_dp_size * inter_dp_size;
  // constexpr int elems_per_thd = ACCESS_BYTES / sizeof(T);
  // int elems_per_blk = blockDim.x * elems_per_thd;
  // int const& tid = threadIdx.x;
  // int const& bid = blockIdx.x;

  // // Divide inputs and outputs into blocks
  // size_t max_num_elems_in;
  // if constexpr (is_opt_waits_mode(static_cast<MixedCommMode>(mode))) {
  //   max_num_elems_in = round_up<elems_per_thd>(ceil_div(num_elems_in_all, gridDim.x));
  // } else if constexpr (is_opt_bytes_mode(static_cast<MixedCommMode>(mode))) {
  //   max_num_elems_in = round_up(ceil_div(num_elems_in_all, gridDim.x),
  //                               elems_per_thd * local_tp_size * inter_tp_size);
  // } else {
  //   max_num_elems_in =
  //       round_up<elems_per_thd * local_tp_size>(ceil_div(num_elems_in_all, gridDim.x));
  // }
  // size_t ofst_elems_in = bid * max_num_elems_in;
  // if (ofst_elems_in >= num_elems_in_all) {
  //   return;
  // }
  // size_t num_elems_in = min(num_elems_in_all - ofst_elems_in, max_num_elems_in);
  // size_t num_elems_out = dp_size * num_elems_in;
  // T* x_out = x_out_all + ofst_elems_in;
  // T const* x_in = x_in_all + ofst_elems_in;
  // T reg_data[elems_per_thd];

  // if constexpr (is_opt_waits_mode(static_cast<MixedCommMode>(mode))) {
  //   // Max virtual memory buffer size:
  //   //   max_local_bytes * inter_dp_size * local_size
  //   // Max nvshmem buffer size:
  //   //   max_local_bytes * num_nodes

  //   // Divide virtual memory and nvshmem buffers into blocks
  //   size_t num_elems_mid = inter_dp_size * num_elems_in;
  //   size_t ofst_data_local = local_size * inter_dp_size * ofst_elems_in;
  //   T* mem_data = mem_data_all + ofst_data_local;
  //   T* mc_data =
  //       mc_data_full_all +
  //       (ofst_data_local + (local_dp_rank + local_tp_rank * local_dp_size) * num_elems_mid);
  //   T* ns_data = ns_data_all + num_nodes * ofst_elems_in;
  //   T* ns_data_self = ns_data + (inter_dp_rank + inter_tp_rank * inter_dp_size) * num_elems_in;
  //   uint32_t* mem_signal = mem_signal_all + bid;
  //   uint32_t* mc_signal = mc_signal_full_all + bid;
  //   uint64_t* ns_signal = ns_signal_all + bid;

  //   // Read inputs
  //   for (size_t i = tid * elems_per_thd; i < num_elems_in; i += elems_per_blk) {
  //     copy_data(ns_data_self + i, x_in + i);
  //   }
  //   __syncthreads();
  //   // Gather data across different nodes
  //   size_t ns_msg_elems_in = floor_div<elems_per_thd>(num_elems_in);
  //   if (tid < num_nodes && tid != node_id) {
  //     int peer_world_rank = local_rank + tid * local_size;
  //     nvshmem_put128_signal_nbi(ns_data_self, ns_data_self, ns_msg_elems_in, ns_signal, 1,
  //                               NVSHMEM_SIGNAL_ADD, peer_world_rank);
  //   }
  //   // Wait for gathered data to be ready
  //   if (tid == 0) {
  //     nvshmem_signal_wait_until(ns_signal, NVSHMEM_CMP_EQ, num_nodes - 1);
  //   }
  //   __syncthreads();

  //   if (inter_tp_size > 1) {
  //     for (size_t i = tid * elems_per_thd; i < num_elems_mid; i += elems_per_blk) {
  //       // Reduce data across different nodes
  //       signal_mode_reduce(reg_data, ns_data + i, num_elems_mid, inter_tp_size);
  //       // Gather data in the same node
  //       multimem_store(mc_data + i, reg_data);
  //     }
  //   } else {
  //     // Gather data in the same node
  //     for (size_t i = tid * elems_per_thd; i < num_elems_mid; i += elems_per_blk) {
  //       copy_data(reg_data, ns_data + i);
  //       multimem_store(mc_data + i, reg_data);
  //     }
  //   }
  //   __syncthreads();
  //   // Wait for gathered data to be ready
  //   multimem_sync(mc_signal, mem_signal, local_size, tid);

  //   for (size_t i = tid * elems_per_thd; i < num_elems_out; i += elems_per_blk) {
  //     int idx_outer_raw = i / num_elems_in;
  //     int idx_outer_local = idx_outer_raw / inter_dp_size;
  //     int idx_outer_inter = idx_outer_raw - idx_outer_local * inter_dp_size;
  //     int idx_outer = idx_outer_local + idx_outer_inter * local_dp_size;
  //     size_t idx_inner = i - idx_outer_raw * num_elems_in;
  //     T* x_out_val = x_out + (idx_outer * num_elems_in_all + idx_inner);
  //     if constexpr (local_tp_size > 1) {
  //       // Reduce data in the same node
  //       signal_mode_reduce<local_tp_size>(reg_data, mem_data + i, num_elems_out);
  //       // Write outputs
  //       copy_data(x_out_val, reg_data);
  //     } else {
  //       // Write outputs
  //       copy_data(x_out_val, mem_data + i);
  //     }
  //   }
  //   // Reset signals
  //   if (tid == 0) {
  //     *mem_signal = 0;
  //     *ns_signal = 0;
  //   }
  // } else if constexpr (is_opt_bytes_mode(static_cast<MixedCommMode>(mode))) {
  //   // Max virtual memory buffer size:
  //   //   max_local_bytes * min(inter_dp_size * local_size, dp_size + 1)
  //   // Max nvshmem buffer size:
  //   //   max_local_bytes / local_tp_size * (inter_dp_size + 2)

  //   // Divide virtual memory and nvshmem buffers into blocks
  //   size_t num_elems_in_chunk = floor_div<local_tp_size>(max_num_elems_in);
  //   size_t ofst_elems_in_chunk = local_tp_rank * num_elems_in_chunk;
  //   size_t num_elems_mid_chunk = num_elems_in_chunk / inter_tp_size;
  //   size_t ofst_elems_mid_chunk = inter_tp_rank * num_elems_mid_chunk;
  //   size_t num_elems_mid = inter_dp_size * num_elems_in_chunk;
  //   T *mem_data_in, *mc_data_in, *mem_data_out, *mc_data_out;
  //   if constexpr (local_tp_size > 1) {
  //     size_t ofst_data_local_out = max_num_elems_in * gridDim.x + dp_size * ofst_elems_in;
  //     mem_data_in = mem_data_all + ofst_elems_in;
  //     mc_data_in = mc_data_tp_all + (ofst_elems_in + ofst_elems_in_chunk);
  //     mem_data_out = mem_data_all + ofst_data_local_out;
  //     mc_data_out = mc_data_full_all +
  //                   (ofst_data_local_out + local_dp_rank * max_num_elems_in +
  //                   ofst_elems_in_chunk);
  //   } else {
  //     size_t ofst_data_local_out = dp_size * ofst_elems_in;
  //     mem_data_out = mem_data_all + ofst_data_local_out;
  //     mc_data_out = mc_data_full_all + (ofst_data_local_out + local_dp_rank * max_num_elems_in);
  //   }
  //   int bid_2 = bid * 2;
  //   uint32_t *mem_signal_in, *mc_signal_in, *mem_signal_out, *mc_signal_out;
  //   if constexpr (local_tp_size > 1) {
  //     mem_signal_in = mem_signal_all + bid_2;
  //     mc_signal_in = mc_signal_tp_all + bid_2;
  //     mem_signal_out = mem_signal_in + 1;
  //     mc_signal_out = mc_signal_full_all + (bid_2 + 1);
  //   } else {
  //     mem_signal_out = mem_signal_all + bid;
  //     mc_signal_out = mc_signal_full_all + bid;
  //   }
  //   T *ns_data_in, *ns_data_mid, *ns_data_out, *ns_data_out_self;
  //   uint64_t *ns_signal_in, *ns_signal_out;
  //   if (inter_tp_size > 1) {
  //     ns_data_in = ns_data_all + bid * (inter_dp_size + 2) * num_elems_in_chunk;
  //     ns_data_mid = ns_data_in + num_elems_in_chunk;
  //     ns_data_out = ns_data_mid + num_elems_in_chunk;
  //     ns_data_out_self = ns_data_out + node_id * num_elems_mid_chunk;
  //     ns_signal_in = ns_signal_all + bid_2;
  //     ns_signal_out = ns_signal_in + 1;
  //   } else {
  //     ns_data_out = ns_data_all + bid * num_elems_mid;
  //     ns_data_out_self = ns_data_out + inter_dp_rank * num_elems_in_chunk;
  //     ns_signal_out = ns_signal_all + bid;
  //   }

  //   if constexpr (local_tp_size > 1) {
  //     // Read inputs
  //     for (size_t i = tid * elems_per_thd; i < num_elems_in; i += elems_per_blk) {
  //       copy_data(mem_data_in + i, x_in + i);
  //     }
  //     __syncthreads();
  //     // Wait for inputs to be ready
  //     multimem_sync<false>(mc_signal_in, mem_signal_in, local_tp_size, tid);
  //     // Reduce data in the same node
  //     if (inter_tp_size > 1) {
  //       for (size_t i = tid * elems_per_thd; i < num_elems_in_chunk; i += elems_per_blk) {
  //         int peer_inter_tp_rank = i / num_elems_mid_chunk;
  //         T* dst = (peer_inter_tp_rank == inter_tp_rank) ? ns_data_mid : ns_data_in;
  //         multimem_load_reduce(reg_data, mc_data_in + i);
  //         copy_data(dst + i, reg_data);
  //       }
  //     } else {
  //       for (size_t i = tid * elems_per_thd; i < num_elems_in_chunk; i += elems_per_blk) {
  //         multimem_load_reduce(reg_data, mc_data_in + i);
  //         copy_data(ns_data_out_self + i, reg_data);
  //       }
  //     }
  //   } else {
  //     // Read inputs
  //     if (inter_tp_size > 1) {
  //       for (size_t i = tid * elems_per_thd; i < num_elems_in; i += elems_per_blk) {
  //         int peer_inter_tp_rank = i / num_elems_mid_chunk;
  //         T* dst = (peer_inter_tp_rank == inter_tp_rank) ? ns_data_mid : ns_data_in;
  //         copy_data(dst + i, x_in + i);
  //       }
  //     } else {
  //       for (size_t i = tid * elems_per_thd; i < num_elems_in; i += elems_per_blk) {
  //         copy_data(ns_data_out_self + i, x_in + i);
  //       }
  //     }
  //   }
  //   __syncthreads();

  //   size_t ns_msg_elems_mid_chunk = floor_div<elems_per_thd>(num_elems_mid_chunk);
  //   if (inter_tp_size > 1) {
  //     // Gather data across different nodes
  //     if (tid < inter_tp_size && tid != inter_tp_rank) {
  //       int peer_world_rank = local_rank + (tid + inter_dp_rank * inter_tp_size) * local_size;
  //       T* dst = ns_data_mid + inter_tp_rank * num_elems_mid_chunk;
  //       T const* src = ns_data_in + tid * num_elems_mid_chunk;
  //       nvshmem_put128_signal_nbi(dst, src, ns_msg_elems_mid_chunk, ns_signal_in, 1,
  //                                 NVSHMEM_SIGNAL_ADD, peer_world_rank);
  //     }
  //     // Wait for gathered data to be ready
  //     if (tid == 0) {
  //       nvshmem_signal_wait_until(ns_signal_in, NVSHMEM_CMP_EQ, inter_tp_size - 1);
  //     }
  //     __syncthreads();
  //     // Reduce data across different nodes
  //     for (size_t i = tid * elems_per_thd; i < num_elems_mid_chunk; i += elems_per_blk) {
  //       signal_mode_reduce(reg_data, ns_data_mid + i, num_elems_mid_chunk, inter_tp_size);
  //       copy_data(ns_data_out_self + i, reg_data);
  //     }
  //     __syncthreads();
  //   }
  //   // Gather data across different nodes
  //   if (tid < num_nodes && tid != node_id) {
  //     int peer_world_rank = local_rank + tid * local_size;
  //     nvshmem_put128_signal_nbi(ns_data_out_self, ns_data_out_self, ns_msg_elems_mid_chunk,
  //                               ns_signal_out, 1, NVSHMEM_SIGNAL_ADD, peer_world_rank);
  //   }
  //   // Wait for gathered data to be ready
  //   if (tid == 0) {
  //     nvshmem_signal_wait_until(ns_signal_out, NVSHMEM_CMP_EQ, num_nodes - 1);
  //   }
  //   __syncthreads();

  //   // Gather data in the same node
  //   size_t local_step = local_dp_size * max_num_elems_in;
  //   for (size_t i = tid * elems_per_thd; i < num_elems_mid; i += elems_per_blk) {
  //     int idx_outer = i / num_elems_in_chunk;
  //     size_t idx_inner = i - idx_outer * num_elems_in_chunk;
  //     copy_data(reg_data, ns_data_out + i);
  //     multimem_store(mc_data_out + (idx_outer * local_step + idx_inner), reg_data);
  //   }
  //   __syncthreads();
  //   // Wait for gathered data to be ready
  //   multimem_sync(mc_signal_out, mem_signal_out, local_size, tid);
  //   // Write outputs
  //   for (size_t i = tid * elems_per_thd; i < num_elems_out; i += elems_per_blk) {
  //     int idx_outer = i / num_elems_in;
  //     size_t idx_inner = i - idx_outer * num_elems_in;
  //     copy_data(x_out + (idx_outer * num_elems_in_all + idx_inner),
  //               mem_data_out + (idx_outer * max_num_elems_in + idx_inner));
  //   }
  //   // Reset signals
  //   if (tid == 0) {
  //     if constexpr (local_tp_size > 1) {
  //       *reinterpret_cast<uint64_t*>(mem_signal_in) = 0;
  //     } else {
  //       *mem_signal_out = 0;
  //     }
  //     if (inter_tp_size > 1) {
  //       *reinterpret_cast<__uint128_t*>(ns_signal_in) = 0;
  //     } else {
  //       *ns_signal_out = 0;
  //     }
  //   }
  // } else {
  //   static_assert(is_mixed_mode(static_cast<MixedCommMode>(mode)), "Unsupported mode");
  //   // Max virtual memory buffer size:
  //   //   max_local_bytes * min(inter_dp_size * local_size, dp_size + 1)
  //   // Max nvshmem buffer size:
  //   //   max_local_bytes / local_tp_size * num_nodes

  //   // Divide virtual memory and nvshmem buffers into blocks
  //   size_t num_elems_in_chunk = floor_div<local_tp_size>(max_num_elems_in);
  //   size_t ofst_elems_in_chunk = local_tp_rank * num_elems_in_chunk;
  //   size_t num_elems_mid = inter_dp_size * num_elems_in_chunk;
  //   T *mem_data_in, *mc_data_in, *mem_data_out, *mc_data_out;
  //   if constexpr (local_tp_size > 1) {
  //     size_t ofst_data_local_out = max_num_elems_in * gridDim.x + dp_size * ofst_elems_in;
  //     mem_data_in = mem_data_all + ofst_elems_in;
  //     mc_data_in = mc_data_tp_all + (ofst_elems_in + ofst_elems_in_chunk);
  //     mem_data_out = mem_data_all + ofst_data_local_out;
  //     mc_data_out = mc_data_full_all +
  //                   (ofst_data_local_out + local_dp_rank * max_num_elems_in +
  //                   ofst_elems_in_chunk);
  //   } else {
  //     size_t ofst_data_local_out = dp_size * ofst_elems_in;
  //     mem_data_out = mem_data_all + ofst_data_local_out;
  //     mc_data_out = mc_data_full_all + (ofst_data_local_out + local_dp_rank * max_num_elems_in);
  //   }
  //   T* ns_data = ns_data_all + bid * num_nodes * num_elems_in_chunk;
  //   T* ns_data_self =
  //       ns_data + (inter_dp_rank + inter_tp_rank * inter_dp_size) * num_elems_in_chunk;
  //   uint32_t *mem_signal_in, *mc_signal_in, *mem_signal_out, *mc_signal_out;
  //   if constexpr (local_tp_size > 1) {
  //     int bid_2 = bid * 2;
  //     mem_signal_in = mem_signal_all + bid_2;
  //     mc_signal_in = mc_signal_tp_all + bid_2;
  //     mem_signal_out = mem_signal_in + 1;
  //     mc_signal_out = mc_signal_full_all + (bid_2 + 1);
  //   } else {
  //     mem_signal_out = mem_signal_all + bid;
  //     mc_signal_out = mc_signal_full_all + bid;
  //   }
  //   uint64_t* ns_signal = ns_signal_all + bid;

  //   if constexpr (local_tp_size > 1) {
  //     // Read inputs
  //     for (size_t i = tid * elems_per_thd; i < num_elems_in; i += elems_per_blk) {
  //       copy_data(mem_data_in + i, x_in + i);
  //     }
  //     __syncthreads();
  //     // Wait for inputs to be ready
  //     multimem_sync<false>(mc_signal_in, mem_signal_in, local_tp_size, tid);
  //     // Reduce data in the same node
  //     for (size_t i = tid * elems_per_thd; i < num_elems_in_chunk; i += elems_per_blk) {
  //       multimem_load_reduce(reg_data, mc_data_in + i);
  //       copy_data(ns_data_self + i, reg_data);
  //     }
  //   } else {
  //     // Read inputs
  //     for (size_t i = tid * elems_per_thd; i < num_elems_in; i += elems_per_blk) {
  //       copy_data(ns_data_self + i, x_in + i);
  //     }
  //   }
  //   __syncthreads();

  //   // Gather data across different nodes
  //   size_t ns_msg_elems_in_chunk = floor_div<elems_per_thd>(num_elems_in_chunk);
  //   if (tid < num_nodes && tid != node_id) {
  //     int peer_world_rank = local_rank + tid * local_size;
  //     nvshmem_put128_signal_nbi(ns_data_self, ns_data_self, ns_msg_elems_in_chunk, ns_signal, 1,
  //                               NVSHMEM_SIGNAL_ADD, peer_world_rank);
  //   }
  //   // Wait for gathered data to be ready
  //   if (tid == 0) {
  //     nvshmem_signal_wait_until(ns_signal, NVSHMEM_CMP_EQ, num_nodes - 1);
  //   }
  //   __syncthreads();

  //   size_t local_step = local_dp_size * max_num_elems_in;
  //   if (inter_tp_size > 1) {
  //     for (size_t i = tid * elems_per_thd; i < num_elems_mid; i += elems_per_blk) {
  //       int idx_outer = i / num_elems_in_chunk;
  //       size_t idx_inner = i - idx_outer * num_elems_in_chunk;
  //       // Reduce data across different nodes
  //       signal_mode_reduce(reg_data, ns_data + i, num_elems_mid, inter_tp_size);
  //       // Gather data in the same node
  //       multimem_store(mc_data_out + (idx_outer * local_step + idx_inner), reg_data);
  //     }
  //   } else {
  //     // Gather data in the same node
  //     for (size_t i = tid * elems_per_thd; i < num_elems_mid; i += elems_per_blk) {
  //       int idx_outer = i / num_elems_in_chunk;
  //       size_t idx_inner = i - idx_outer * num_elems_in_chunk;
  //       copy_data(reg_data, ns_data + i);
  //       multimem_store(mc_data_out + (idx_outer * local_step + idx_inner), reg_data);
  //     }
  //   }
  //   __syncthreads();
  //   // Wait for gathered data to be ready
  //   multimem_sync(mc_signal_out, mem_signal_out, local_size, tid);
  //   // Write outputs
  //   for (size_t i = tid * elems_per_thd; i < num_elems_out; i += elems_per_blk) {
  //     int idx_outer = i / num_elems_in;
  //     size_t idx_inner = i - idx_outer * num_elems_in;
  //     copy_data(x_out + (idx_outer * num_elems_in_all + idx_inner),
  //               mem_data_out + (idx_outer * max_num_elems_in + idx_inner));
  //   }
  //   // Reset signals
  //   if (tid == 0) {
  //     if constexpr (local_tp_size > 1) {
  //       *reinterpret_cast<uint64_t*>(mem_signal_in) = 0;
  //     } else {
  //       *mem_signal_out = 0;
  //     }
  //     *ns_signal = 0;
  //   }
  // }
}

template <int local_tp_size, int local_dp_size, int mode_val, typename T>
__global__ void fused_reducescatter_allreduce_single_node(MixedCommArgs<T> args) {
  constexpr MixedCommMode mode = static_cast<MixedCommMode>(mode_val);
  constexpr int local_size = local_tp_size * local_dp_size;
  int local_rank = args.local_tp_rank + args.local_dp_rank * local_tp_size;
  constexpr int elems_per_thd = ACCESS_BYTES / sizeof(T);
  int elems_per_blk = blockDim.x * elems_per_thd;
  int const& tid = threadIdx.x;
  int const& bid = blockIdx.x;
  int const ofst_index = tid * elems_per_thd;
  int const bid_2 = bid * 2;

  // Divide inputs and outputs into blocks
  size_t max_num_elems_out;
  if constexpr (is_opt_waits_mode(mode)) {
    max_num_elems_out = round_up<elems_per_thd>(ceil_div(args.num_elems_out, gridDim.x));
  } else {
    max_num_elems_out =
        round_up<elems_per_thd * local_tp_size>(ceil_div(args.num_elems_out, gridDim.x));
  }
  size_t ofst_elems_out = bid * max_num_elems_out;
  size_t num_elems_out = (args.num_elems_out > ofst_elems_out)
                             ? min(args.num_elems_out - ofst_elems_out, max_num_elems_out)
                             : 0;
  T* x_out = args.x_out + ofst_elems_out;
  T const* x_in = args.x_in + ofst_elems_out;

  // Prepare basic variables
  constexpr bool vm_use_lamport = single_node_vm_use_lamport(mode);
  T reg_data[elems_per_thd];
  IndexInfo index_info;
  index_info.raw = args.index_info[bid];
  int vm_index = get_vm_index<vm_use_lamport>(index_info);

  if constexpr (is_opt_waits_mode(mode) || local_tp_size == 1) {
    // Max virtual memory buffer size:
    //   max_local_bytes * local_size
    constexpr bool use_multicast_load_reduce =
        is_mc_mode(mode) && local_tp_size == 1 && !vm_use_lamport;
    constexpr bool use_multicast_store = is_mc_mode(mode) && local_dp_size == 1;
    size_t ofst_data = local_size * ofst_elems_out;
    size_t ofst_data_self = ofst_data + local_rank * max_num_elems_out;

    if constexpr (use_multicast_load_reduce) {
      // Read inputs
      T* mem_data = args.mem_data_buffer[vm_index] + ofst_data;
      for (size_t i = ofst_index; i < num_elems_out; i += elems_per_blk) {
        T* dst = mem_data + i;
        T const* src = x_in + i;
#pragma unroll
        for (int j = 0; j < local_dp_size; ++j) {
          copy_data(dst, src);
          dst += max_num_elems_out;
          src += args.num_elems_out;
        }
      }
      T* mc_data = args.mc_data_full_buffer[vm_index] + ofst_data_self;
      __syncthreads();
      if (tid == 0) {
        update_index_info_vm<vm_use_lamport>(index_info);
        args.index_info[bid] = index_info.raw;
        uint32_t* mem_signal = args.mem_signal_buffer[vm_index] + bid;
        uint32_t* mc_signal = args.mc_signal_full_buffer[vm_index] + bid;
        multicast_sync<local_size, /*use_release=*/false>(mc_signal, mem_signal);
        *mem_signal = 0;
      }
      __syncthreads();
      // Reduce data and write outputs
      for (size_t i = ofst_index; i < num_elems_out; i += elems_per_blk) {
        multicast_load_reduce(reg_data, mc_data + i);
        copy_data(x_out + i, reg_data);
      }
    } else {
      // Read inputs and gather data
      T *mc_data, *uc_data_array[local_size];
      if constexpr (use_multicast_store) {
        mc_data = args.mc_data_full_buffer[vm_index] + ofst_data_self;
      } else {
        set_uc_data_array<local_tp_size, local_dp_size>(
            uc_data_array, args.uc_data_array_buffer + vm_index * local_size, args.local_tp_rank,
            args.local_dp_rank, ofst_data_self);
      }
      size_t ofst_src_base = args.local_dp_rank * args.num_elems_out;
      for (size_t i = ofst_index; i < num_elems_out; i += elems_per_blk) {
#pragma unroll
        for (int j = 0; j < local_dp_size; ++j) {
          size_t ofst_src = ofst_src_base + i;
          copy_data<T, vm_use_lamport>(reg_data, x_in + ofst_src);
          if constexpr (use_multicast_store) {
            multicast_store(mc_data + i, reg_data);
          } else {
            unicast_store<local_tp_size>(uc_data_array + j * local_tp_size, reg_data, i);
          }
          ofst_src_base += args.num_elems_out;
          if (ofst_src_base == args.num_elems_in) {
            ofst_src_base = 0;
          }
        }
      }
      T* mem_data;
      if constexpr (use_multicast_store) {
        mem_data = args.mem_data_buffer[vm_index] + ofst_data;
      } else {
        mem_data = uc_data_array[0] - ofst_data_self + ofst_data;
      }
      if constexpr (!vm_use_lamport) {
        __syncthreads();
        if (tid == 0) {
          update_index_info_vm<vm_use_lamport>(index_info);
          args.index_info[bid] = index_info.raw;
          uint32_t* mem_signal = args.mem_signal_buffer[vm_index] + bid;
          uint32_t* mc_signal = args.mc_signal_full_buffer[vm_index] + bid;
          multicast_sync<local_size>(mc_signal, mem_signal);
          *mem_signal = 0;
        }
        __syncthreads();
      } else {
        int4* reset_info = args.reset_info + bid_2;
        ResetInfo<T> vm_reset;
        vm_reset.raw = reset_info[0];
        for (size_t i = ofst_index; i < vm_reset.info.size; i += elems_per_blk) {
          fill_neg_zero(vm_reset.info.data + i);
        }
        __syncthreads();
        if (tid == 0) {
          update_index_info_vm<vm_use_lamport>(index_info);
          args.index_info[bid] = index_info.raw;
          vm_reset.info.data = mem_data;
          vm_reset.info.size = local_size * max_num_elems_out;
          reset_info[0] = vm_reset.raw;
        }
      }
      // Reduce data and write outputs
      for (size_t i = ofst_index; i < num_elems_out; i += elems_per_blk) {
        reduce_data<local_size, vm_use_lamport>(reg_data, mem_data + i, max_num_elems_out);
        copy_data(x_out + i, reg_data);
      }
    }
  } else {
    static_assert(is_opt_bytes_mode(mode) && local_tp_size > 1, "Unsupported mode");
    // Max virtual memory buffer size:
    //   max_local_bytes * min(local_size, local_dp_size + 1)
    constexpr bool use_multicast = is_mc_mode(mode);
    size_t num_elems_out_chunk = floor_div<local_tp_size>(max_num_elems_out);
    size_t ofst_elems_out_chunk = args.local_tp_rank * num_elems_out_chunk;
    size_t ofst_data_in = local_dp_size * ofst_elems_out;
    size_t ofst_data_in_self =
        ofst_data_in + args.local_dp_rank * max_num_elems_out + ofst_elems_out_chunk;
    size_t num_elems_mem_data_in = local_dp_size * max_num_elems_out;
    size_t ofst_data_out = num_elems_mem_data_in * gridDim.x + ofst_elems_out;
    size_t ofst_data_out_self = ofst_data_out + ofst_elems_out_chunk;

    if constexpr (use_multicast && !vm_use_lamport) {
      // Read inputs
      T* mem_data_all = args.mem_data_buffer[vm_index];
      T* mem_data_in = mem_data_all + ofst_data_in;
      for (size_t i = ofst_index; i < num_elems_out; i += elems_per_blk) {
        T* dst = mem_data_in + i;
        T const* src = x_in + i;
#pragma unroll
        for (int j = 0; j < local_dp_size; ++j) {
          copy_data(dst, src);
          dst += max_num_elems_out;
          src += args.num_elems_out;
        }
      }
      T* mc_data_in = args.mc_data_full_buffer[vm_index] + ofst_data_in_self;
      T* mc_data_out;
      if constexpr (local_dp_size == 1) {
        mc_data_out = mc_data_in - ofst_data_in_self + ofst_data_out_self;
      } else {
        mc_data_out = args.mc_data_tp_buffer[vm_index] + ofst_data_out_self;
      }
      __syncthreads();
      uint32_t *mem_signal_in, *mc_signal_in;
      if (tid == 0) {
        mem_signal_in = args.mem_signal_buffer[vm_index] + bid_2;
        mc_signal_in = args.mc_signal_full_buffer[vm_index] + bid_2;
        multicast_sync<local_size, /*use_release=*/false>(mc_signal_in, mem_signal_in);
      }
      __syncthreads();
      // Reduce and gather data
      for (size_t i = ofst_index; i < num_elems_out_chunk; i += elems_per_blk) {
        multicast_load_reduce(reg_data, mc_data_in + i);
        multicast_store(mc_data_out + i, reg_data);
      }
      T* mem_data_out = mem_data_all + ofst_data_out;
      __syncthreads();
      if (tid == 0) {
        update_index_info_vm<vm_use_lamport>(index_info);
        args.index_info[bid] = index_info.raw;
        uint32_t* mem_signal_out = mem_signal_in + 1;
        uint32_t* mc_signal_out;
        if constexpr (local_dp_size == 1) {
          mc_signal_out = mc_signal_in + 1;
        } else {
          mc_signal_out = args.mc_signal_tp_buffer[vm_index] + (bid_2 + 1);
        }
        multicast_sync<local_tp_size>(mc_signal_out, mem_signal_out);
        *reinterpret_cast<uint64_t*>(mem_signal_in) = 0;
      }
      __syncthreads();
      // Write outputs
      for (size_t i = ofst_index; i < num_elems_out; i += elems_per_blk) {
        copy_data(x_out + i, mem_data_out + i);
      }
    } else {
      // Read inputs and gather data
      T* uc_data_array[local_size];
      set_uc_data_array<local_tp_size, local_dp_size>(
          uc_data_array, args.uc_data_array_buffer + vm_index * local_size, args.local_tp_rank,
          args.local_dp_rank, ofst_data_in_self);
      T* mem_data_all = uc_data_array[0] - ofst_data_in_self;
      T* mem_data_in = mem_data_all + ofst_data_in;
      size_t ofst_src_dp_base = args.local_dp_rank * args.num_elems_out;
      size_t ofst_src_tp_base = args.local_tp_rank * num_elems_out_chunk;
      for (size_t i = ofst_index; i < num_elems_out_chunk; i += elems_per_blk) {
#pragma unroll
        for (int j = 0; j < local_dp_size; ++j) {
          T const* x_in_base = x_in + ofst_src_dp_base;
#pragma unroll
          for (int k = 0; k < local_tp_size; ++k) {
            size_t ofst_src_tp = ofst_src_tp_base + i;
            if (ofst_src_tp < num_elems_out) {
              copy_data<T, vm_use_lamport>(reg_data, x_in_base + ofst_src_tp);
              copy_data(uc_data_array[j * local_tp_size + k] + i, reg_data);
            }
            ofst_src_tp_base += num_elems_out_chunk;
            if (ofst_src_tp_base == max_num_elems_out) {
              ofst_src_tp_base = 0;
            }
          }
          ofst_src_dp_base += args.num_elems_out;
          if (ofst_src_dp_base == args.num_elems_in) {
            ofst_src_dp_base = 0;
          }
        }
      }
      if constexpr (vm_use_lamport) {
        size_t num_elems_src = (num_elems_out > ofst_src_tp_base)
                                   ? min(num_elems_out - ofst_src_tp_base, num_elems_out_chunk)
                                   : 0;
        for (size_t i = num_elems_src + ofst_index; i < num_elems_out_chunk; i += elems_per_blk) {
          T* mem_data_in_val = mem_data_in + i;
#pragma unroll
          for (int j = 0; j < local_size; ++j) {
            *reinterpret_cast<__uint128_t*>(mem_data_in_val) = 0;
            mem_data_in_val += num_elems_out_chunk;
          }
        }
      }
      T* mc_data_out;
      if constexpr (use_multicast) {
        mc_data_out = args.mc_data_tp_buffer[vm_index] + ofst_data_out_self;
      } else {
        size_t ofst_uc_data = ofst_data_out_self - ofst_data_in_self;
#pragma unroll
        for (int i = 0; i < local_tp_size; ++i) {
          uc_data_array[i] += ofst_uc_data;
        }
      }
      uint32_t *mem_signal_in, *mc_signal_in;
      int4* reset_info;
      ResetInfo<T> vm_reset;
      if constexpr (!vm_use_lamport) {
        __syncthreads();
        if (tid == 0) {
          mem_signal_in = args.mem_signal_buffer[vm_index] + bid_2;
          mc_signal_in = args.mc_signal_full_buffer[vm_index] + bid_2;
          multicast_sync<local_size>(mc_signal_in, mem_signal_in);
        }
        __syncthreads();
      } else {
        reset_info = args.reset_info + bid_2;
        vm_reset.raw = reset_info[0];
        for (size_t i = ofst_index; i < vm_reset.info.size; i += elems_per_blk) {
          fill_neg_zero(vm_reset.info.data + i);
        }
      }
      // Reduce and gather data
      for (size_t i = ofst_index; i < num_elems_out_chunk; i += elems_per_blk) {
        reduce_data<local_size, vm_use_lamport, T, /*use_replace=*/true>(reg_data, mem_data_in + i,
                                                                         num_elems_out_chunk);
        if constexpr (use_multicast) {
          multicast_store(mc_data_out + i, reg_data);
        } else {
          unicast_store<local_tp_size>(uc_data_array, reg_data, i);
        }
      }
      T* mem_data_out = mem_data_all + ofst_data_out;
      __syncthreads();
      if constexpr (!vm_use_lamport) {
        if (tid == 0) {
          update_index_info_vm<vm_use_lamport>(index_info);
          args.index_info[bid] = index_info.raw;
          uint32_t* mem_signal_out = mem_signal_in + 1;
          uint32_t* mc_signal_out;
          if constexpr (local_dp_size == 1) {
            mc_signal_out = mc_signal_in + 1;
          } else {
            mc_signal_out = args.mc_signal_tp_buffer[vm_index] + (bid_2 + 1);
          }
          multicast_sync<local_tp_size>(mc_signal_out, mem_signal_out);
          *reinterpret_cast<uint64_t*>(mem_signal_in) = 0;
        }
        __syncthreads();
      } else {
        for (size_t i = ofst_index; i < num_elems_mem_data_in; i += elems_per_blk) {
          fill_neg_zero(mem_data_in + i);
        }
        if (tid == 0) {
          update_index_info_vm<vm_use_lamport>(index_info);
          args.index_info[bid] = index_info.raw;
          vm_reset.info.data = mem_data_out;
          vm_reset.info.size = max_num_elems_out;
          reset_info[0] = vm_reset.raw;
        }
      }
      // Write outputs
      for (size_t i = ofst_index; i < num_elems_out; i += elems_per_blk) {
        load_data<vm_use_lamport>(reg_data, mem_data_out + i);
        copy_data(x_out + i, reg_data);
      }
    }
  }
}

template <int local_tp_size, int local_dp_size, int mode, typename T>
__global__ void fused_reducescatter_allreduce_multi_node(MixedCommArgs<T> args) {
  //   constexpr int local_size = local_tp_size * local_dp_size;
  //   int local_rank = local_tp_rank + local_dp_rank * local_tp_size;
  //   int num_nodes = inter_tp_size * inter_dp_size;
  //   int node_id = inter_tp_rank + inter_dp_rank * inter_tp_size;
  //   int dp_size = local_dp_size * inter_dp_size;
  //   constexpr int elems_per_thd = ACCESS_BYTES / sizeof(T);
  //   int elems_per_blk = blockDim.x * elems_per_thd;
  //   int const& tid = threadIdx.x;
  //   int const& bid = blockIdx.x;

  //   // Divide inputs and outputs into blocks
  //   size_t max_num_elems_out;
  //   if constexpr (is_opt_waits_mode(static_cast<MixedCommMode>(mode))) {
  //     max_num_elems_out = round_up<elems_per_thd>(ceil_div(num_elems_out_all, gridDim.x));
  //   } else if constexpr (is_opt_bytes_mode(static_cast<MixedCommMode>(mode))) {
  //     max_num_elems_out = round_up(ceil_div(num_elems_out_all, gridDim.x),
  //                                  elems_per_thd * local_tp_size * inter_tp_size);
  //   } else {
  //     max_num_elems_out =
  //         round_up<elems_per_thd * local_tp_size>(ceil_div(num_elems_out_all, gridDim.x));
  //   }
  //   size_t ofst_elems_out = bid * max_num_elems_out;
  //   if (ofst_elems_out >= num_elems_out_all) {
  //     return;
  //   }
  //   size_t num_elems_out = min(num_elems_out_all - ofst_elems_out, max_num_elems_out);
  //   size_t num_elems_in = dp_size * num_elems_out;
  //   T* x_out = x_out_all + ofst_elems_out;
  //   T const* x_in = x_in_all + ofst_elems_out;
  //   T reg_data[elems_per_thd];

  //   if constexpr (is_opt_waits_mode(static_cast<MixedCommMode>(mode))) {
  //     // Max virtual memory buffer size:
  //     //   max_local_bytes * inter_dp_size * local_size
  //     // Max nvshmem buffer size:
  //     //   max_local_bytes * (inter_dp_size + num_nodes)

  //     // Divide virtual memory and nvshmem buffers into blocks
  //     size_t num_elems_mid = inter_dp_size * num_elems_out;
  //     size_t ofst_data_local = local_size * inter_dp_size * ofst_elems_out;
  //     size_t ofst_data_local_self = ofst_data_local + local_rank * num_elems_mid;
  //     T* mem_data = mem_data_all + ofst_data_local;
  //     T* mc_data = mc_data_full_all + ofst_data_local_self;
  //     T* uc_data_array[local_size];
  //     if constexpr (local_tp_size > 1) {
  // #pragma unroll
  //       for (int i = 0; i < local_size; ++i) {
  //         uc_data_array[i] = uc_data_all_array[i] + ofst_data_local_self;
  //       }
  //     }
  //     T* ns_data_in = ns_data_all + (inter_dp_size + num_nodes) * ofst_elems_out;
  //     T* ns_data_out = ns_data_in + inter_dp_size * num_elems_out;
  //     T* ns_data_out_self = ns_data_out + node_id * num_elems_out;
  //     uint32_t* mem_signal = mem_signal_all + bid;
  //     uint32_t* mc_signal = mc_signal_full_all + bid;
  //     uint64_t* ns_signal = ns_signal_all + bid;

  //     for (size_t i = tid * elems_per_thd; i < num_elems_in; i += elems_per_blk) {
  //       int idx_outer_in = i / num_elems_out;
  //       size_t idx_inner = i - idx_outer_in * num_elems_out;
  //       T const* x_in_val = x_in + (idx_outer_in * num_elems_out_all + idx_inner);
  //       int idx_outer_out_inter = idx_outer_in / local_dp_size;
  //       int idx_outer_out_local = idx_outer_in - idx_outer_out_inter * local_dp_size;
  //       if constexpr (local_tp_size > 1) {
  //         // Read inputs
  //         copy_data(reg_data, x_in_val);
  //         // Gather data
  //         int peer_local_rank_base = idx_outer_out_local * local_tp_size;
  //         size_t ofst_uc_data = idx_outer_out_inter * num_elems_out + idx_inner;
  // #pragma unroll
  //         for (int j = 0; j < local_tp_size; ++j) {
  //           int peer_local_rank = peer_local_rank_base + mod<local_tp_size>(local_tp_rank + j);
  //           copy_data(uc_data_array[peer_local_rank] + ofst_uc_data, reg_data);
  //         }
  //       } else {
  //         // Read inputs
  //         int idx_outer_out = idx_outer_out_inter + idx_outer_out_local * inter_dp_size;
  //         copy_data(mem_data + (idx_outer_out * num_elems_out + idx_inner), x_in_val);
  //       }
  //     }
  //     __syncthreads();
  //     // Wait for inputs or gathered data to be ready
  //     multimem_sync(mc_signal, mem_signal, local_size, tid);
  //     // Reduce data in the same node
  //     for (size_t i = tid * elems_per_thd; i < num_elems_mid; i += elems_per_blk) {
  //       int idx_outer = i / num_elems_out;
  //       size_t idx_inner = i - idx_outer * num_elems_out;
  //       T* ns_data_val = (idx_outer == inter_dp_rank) ? (ns_data_out_self + idx_inner) :
  //       (ns_data_in + i); if constexpr (local_tp_size > 1) {
  //         signal_mode_reduce<local_size>(reg_data, mem_data + i, num_elems_mid);
  //       } else {
  //         multimem_load_reduce(reg_data, mc_data + i);
  //       }
  //       copy_data(ns_data_val, reg_data);
  //     }
  //     __syncthreads();
  //     // Gather data across different nodes
  //     size_t ns_msg_elems_out = floor_div<elems_per_thd>(num_elems_out);
  //     if (tid < num_nodes && tid != node_id) {
  //       int peer_world_rank = local_rank + tid * local_size;
  //       int peer_inter_dp_rank = tid / inter_tp_size;
  //       T const* src = (peer_inter_dp_rank == inter_dp_rank)
  //                          ? ns_data_out_self
  //                          : (ns_data_in + peer_inter_dp_rank * num_elems_out);
  //       nvshmem_put128_signal_nbi(ns_data_out_self, src, ns_msg_elems_out, ns_signal, 1,
  //                                 NVSHMEM_SIGNAL_ADD, peer_world_rank);
  //     }
  //     // Wait for gathered data to be ready
  //     if (tid == 0) {
  //       nvshmem_signal_wait_until(ns_signal, NVSHMEM_CMP_EQ, num_nodes - 1);
  //     }
  //     __syncthreads();

  //     for (size_t i = tid * elems_per_thd; i < num_elems_out; i += elems_per_blk) {
  //       // Reduce data across different nodes
  //       signal_mode_reduce(reg_data, ns_data_out + i, num_elems_out, num_nodes);
  //       // Write outputs
  //       copy_data(x_out + i, reg_data);
  //     }
  //     // Reset signals
  //     if (tid == 0) {
  //       *mem_signal = 0;
  //       *ns_signal = 0;
  //     }
  //   } else if constexpr (is_opt_bytes_mode(static_cast<MixedCommMode>(mode))) {
  //     // Max virtual memory buffer size:
  //     //   max_local_bytes * min(inter_dp_size * local_size, dp_size + 1)
  //     // Max nvshmem buffer size:
  //     //   max_local_bytes / local_tp_size * min(num_nodes * 2, inter_dp_size * 2 + 1)

  //     // Divide virtual memory and nvshmem buffers into blocks
  //     size_t num_elems_out_chunk = floor_div<local_tp_size>(max_num_elems_out);
  //     size_t ofst_elems_out_chunk = local_tp_rank * num_elems_out_chunk;
  //     size_t num_elems_mid_chunk = num_elems_out_chunk / inter_tp_size;
  //     size_t ofst_elems_mid_chunk = inter_tp_rank * num_elems_mid_chunk;
  //     size_t num_elems_mid = inter_dp_size * num_elems_out_chunk;
  //     T *mem_data_in, *mc_data_in, *mem_data_out, *mc_data_out;
  //     size_t ofst_data_local_in = dp_size * ofst_elems_out;
  //     if constexpr (local_tp_size > 1) {
  //       size_t ofst_data_local_out = dp_size * max_num_elems_out * gridDim.x + ofst_elems_out;
  //       mem_data_in = mem_data_all + ofst_data_local_in;
  //       mc_data_in = mc_data_full_all +
  //                    (ofst_data_local_in + local_dp_rank * max_num_elems_out +
  //                    ofst_elems_out_chunk);
  //       mem_data_out = mem_data_all + ofst_data_local_out;
  //       mc_data_out = mc_data_tp_all + (ofst_data_local_out + ofst_elems_out_chunk);
  //     } else {
  //       mem_data_in = mem_data_all + ofst_data_local_in;
  //       mc_data_in = mc_data_full_all + (ofst_data_local_in + local_dp_rank * max_num_elems_out);
  //     }
  //     int bid_2 = bid * 2;
  //     uint32_t *mem_signal_in, *mc_signal_in, *mem_signal_out, *mc_signal_out;
  //     if constexpr (local_tp_size > 1) {
  //       mem_signal_in = mem_signal_all + bid_2;
  //       mc_signal_in = mc_signal_full_all + bid_2;
  //       mem_signal_out = mem_signal_in + 1;
  //       mc_signal_out = mc_signal_in + 1;
  //     } else {
  //       mem_signal_in = mem_signal_all + bid;
  //       mc_signal_in = mc_signal_full_all + bid;
  //     }
  //     T *ns_data_in, *ns_data_mid, *ns_data_mid_self, *ns_data_out, *ns_data_out_self;
  //     uint64_t *ns_signal_in, *ns_signal_out;
  //     if (inter_tp_size > 1) {
  //       ns_data_in = ns_data_all + bid * (inter_dp_size * 2 + 1) * num_elems_out_chunk;
  //       ns_data_mid = ns_data_in + num_elems_mid;
  //       ns_data_mid_self = ns_data_mid + node_id * num_elems_mid_chunk;
  //       ns_data_out = ns_data_mid + num_elems_mid;
  //       ns_data_out_self = ns_data_out + inter_tp_rank * num_elems_mid_chunk;
  //       ns_signal_in = ns_signal_all + bid_2;
  //       ns_signal_out = ns_signal_in + 1;
  //     } else {
  //       ns_data_in = ns_data_all + bid_2 * num_elems_mid;
  //       ns_data_mid = ns_data_in + num_elems_mid;
  //       ns_data_mid_self = ns_data_mid + node_id * num_elems_mid_chunk;
  //       ns_signal_in = ns_signal_all + bid;
  //     }

  //     // Read inputs
  //     for (size_t i = tid * elems_per_thd; i < num_elems_in; i += elems_per_blk) {
  //       int idx_outer = i / num_elems_out;
  //       size_t idx_inner = i - idx_outer * num_elems_out;
  //       copy_data(mem_data_in + (idx_outer * max_num_elems_out + idx_inner),
  //                 x_in + (idx_outer * num_elems_out_all + idx_inner));
  //     }
  //     __syncthreads();
  //     // Wait for inputs to be ready
  //     multimem_sync<false>(mc_signal_in, mem_signal_in, local_size, tid);
  //     // Reduce data in the same node
  //     size_t local_step = local_dp_size * max_num_elems_out;
  //     if (inter_tp_size > 1) {
  //       for (size_t i = tid * elems_per_thd; i < num_elems_mid; i += elems_per_blk) {
  //         int idx_outer_dp = i / num_elems_out_chunk;
  //         size_t idx_inner_dp = i - idx_outer_dp * num_elems_out_chunk;
  //         int idx_outer_tp = idx_inner_dp / num_elems_mid_chunk;
  //         size_t idx_inner_tp = idx_inner_dp - idx_outer_tp * num_elems_mid_chunk;
  //         T* ns_data_val = (idx_outer_dp == inter_dp_rank && idx_outer_tp == inter_tp_rank)
  //                              ? (ns_data_mid_self + idx_inner_tp)
  //                              : (ns_data_in + i);
  //         multimem_load_reduce(reg_data, mc_data_in + (idx_outer_dp * local_step +
  //         idx_inner_dp)); copy_data(ns_data_val, reg_data);
  //       }
  //     } else {
  //       for (size_t i = tid * elems_per_thd; i < num_elems_mid; i += elems_per_blk) {
  //         int idx_outer = i / num_elems_out_chunk;
  //         size_t idx_inner = i - idx_outer * num_elems_out_chunk;
  //         T* ns_data_val =
  //             (idx_outer == inter_dp_rank) ? (ns_data_mid_self + idx_inner) : (ns_data_in + i);
  //         multimem_load_reduce(reg_data, mc_data_in + (idx_outer * local_step + idx_inner));
  //         copy_data(ns_data_val, reg_data);
  //       }
  //     }
  //     __syncthreads();
  //     // Gather data across different nodes
  //     size_t ns_msg_elems_mid_chunk = floor_div<elems_per_thd>(num_elems_mid_chunk);
  //     if (tid < num_nodes && tid != node_id) {
  //       int peer_world_rank = local_rank + tid * local_size;
  //       T const* src = ns_data_in + tid * num_elems_mid_chunk;
  //       nvshmem_put128_signal_nbi(ns_data_mid_self, src, ns_msg_elems_mid_chunk, ns_signal_in, 1,
  //                                 NVSHMEM_SIGNAL_ADD, peer_world_rank);
  //     }
  //     // Wait for gathered data to be ready
  //     if (tid == 0) {
  //       nvshmem_signal_wait_until(ns_signal_in, NVSHMEM_CMP_EQ, num_nodes - 1);
  //     }
  //     __syncthreads();

  //     if (inter_tp_size > 1) {
  //       // Reduce data across different nodes
  //       for (size_t i = tid * elems_per_thd; i < num_elems_mid_chunk; i += elems_per_blk) {
  //         signal_mode_reduce(reg_data, ns_data_mid + i, num_elems_mid_chunk, num_nodes);
  //         copy_data(ns_data_out_self + i, reg_data);
  //       }
  //       __syncthreads();
  //       // Gather data across different nodes
  //       if (tid < inter_tp_size && tid != inter_tp_rank) {
  //         int peer_world_rank = local_rank + (tid + inter_dp_rank * inter_tp_size) * local_size;
  //         nvshmem_put128_signal_nbi(ns_data_out_self, ns_data_out_self, ns_msg_elems_mid_chunk,
  //                                   ns_signal_out, 1, NVSHMEM_SIGNAL_ADD, peer_world_rank);
  //       }
  //       // Wait for gathered data to be ready
  //       if (tid == 0) {
  //         nvshmem_signal_wait_until(ns_signal_out, NVSHMEM_CMP_EQ, inter_tp_size - 1);
  //       }
  //       __syncthreads();

  //       if constexpr (local_tp_size > 1) {
  //         // Gather data in the same node
  //         for (size_t i = tid * elems_per_thd; i < num_elems_out_chunk; i += elems_per_blk) {
  //           copy_data(reg_data, ns_data_out + i);
  //           multimem_store(mc_data_out + i, reg_data);
  //         }
  //         __syncthreads();
  //       } else {
  //         // Write outputs
  //         for (size_t i = tid * elems_per_thd; i < num_elems_out; i += elems_per_blk) {
  //           copy_data(x_out + i, ns_data_out + i);
  //         }
  //       }
  //     } else {
  //       if constexpr (local_tp_size > 1) {
  //         for (size_t i = tid * elems_per_thd; i < num_elems_out_chunk; i += elems_per_blk) {
  //           // Reduce data across different nodes
  //           signal_mode_reduce(reg_data, ns_data_mid + i, num_elems_out_chunk, num_nodes);
  //           // Gather data in the same node
  //           multimem_store(mc_data_out + i, reg_data);
  //         }
  //       } else {
  //         for (size_t i = tid * elems_per_thd; i < num_elems_out; i += elems_per_blk) {
  //           // Reduce data across different nodes
  //           signal_mode_reduce(reg_data, ns_data_mid + i, num_elems_out_chunk, num_nodes);
  //           // Write outputs
  //           copy_data(x_out + i, reg_data);
  //         }
  //       }
  //     }

  //     if constexpr (local_tp_size > 1) {
  //       // Wait for gathered data to be ready
  //       multimem_sync(mc_signal_out, mem_signal_out, local_size, tid);
  //       // Write outputs
  //       for (size_t i = tid * elems_per_thd; i < num_elems_out; i += elems_per_blk) {
  //         copy_data(x_out + i, mem_data_out + i);
  //       }
  //     }
  //     // Reset signals
  //     if (tid == 0) {
  //       if constexpr (local_tp_size > 1) {
  //         *reinterpret_cast<uint64_t*>(mem_signal_in) = 0;
  //       } else {
  //         *mem_signal_in = 0;
  //       }
  //       if (inter_tp_size > 1) {
  //         *reinterpret_cast<__uint128_t*>(ns_signal_in) = 0;
  //       } else {
  //         *ns_signal_in = 0;
  //       }
  //     }
  //   } else {
  //     static_assert(is_mixed_mode(static_cast<MixedCommMode>(mode)), "Unsupported mode");
  //     // Max virtual memory buffer size:
  //     //   max_local_bytes * min(inter_dp_size * local_size, dp_size + 1)
  //     // Max nvshmem buffer size:
  //     //   max_local_bytes / local_tp_size * (inter_dp_size + num_nodes)

  //     // Divide virtual memory and nvshmem buffers into blocks
  //     size_t num_elems_out_chunk = floor_div<local_tp_size>(max_num_elems_out);
  //     size_t ofst_elems_out_chunk = local_tp_rank * num_elems_out_chunk;
  //     size_t num_elems_mid = inter_dp_size * num_elems_out_chunk;
  //     T *mem_data_in, *mc_data_in, *mem_data_out, *mc_data_out;
  //     size_t ofst_data_local_in = dp_size * ofst_elems_out;
  //     if constexpr (local_tp_size > 1) {
  //       size_t ofst_data_local_out = dp_size * max_num_elems_out * gridDim.x + ofst_elems_out;
  //       mem_data_in = mem_data_all + ofst_data_local_in;
  //       mc_data_in = mc_data_full_all +
  //                    (ofst_data_local_in + local_dp_rank * max_num_elems_out +
  //                    ofst_elems_out_chunk);
  //       mem_data_out = mem_data_all + ofst_data_local_out;
  //       mc_data_out = mc_data_tp_all + (ofst_data_local_out + ofst_elems_out_chunk);
  //     } else {
  //       mem_data_in = mem_data_all + ofst_data_local_in;
  //       mc_data_in = mc_data_full_all + (ofst_data_local_in + local_dp_rank * max_num_elems_out);
  //     }
  //     T* ns_data_in = ns_data_all + bid * (inter_dp_size + num_nodes) * num_elems_out_chunk;
  //     T* ns_data_out = ns_data_in + num_elems_mid;
  //     T* ns_data_out_self = ns_data_out + node_id * num_elems_out_chunk;
  //     uint32_t *mem_signal_in, *mc_signal_in, *mem_signal_out, *mc_signal_out;
  //     if constexpr (local_tp_size > 1) {
  //       int bid_2 = bid * 2;
  //       mem_signal_in = mem_signal_all + bid_2;
  //       mc_signal_in = mc_signal_full_all + bid_2;
  //       mem_signal_out = mem_signal_in + 1;
  //       mc_signal_out = mc_signal_in + 1;
  //     } else {
  //       mem_signal_in = mem_signal_all + bid;
  //       mc_signal_in = mc_signal_full_all + bid;
  //     }
  //     uint64_t* ns_signal = ns_signal_all + bid;

  //     // Read inputs
  //     for (size_t i = tid * elems_per_thd; i < num_elems_in; i += elems_per_blk) {
  //       int idx_outer = i / num_elems_out;
  //       size_t idx_inner = i - idx_outer * num_elems_out;
  //       copy_data(mem_data_in + (idx_outer * max_num_elems_out + idx_inner),
  //                 x_in + (idx_outer * num_elems_out_all + idx_inner));
  //     }
  //     __syncthreads();
  //     // Wait for inputs to be ready
  //     multimem_sync<false>(mc_signal_in, mem_signal_in, local_size, tid);
  //     // Reduce data in the same node
  //     size_t local_step = local_dp_size * max_num_elems_out;
  //     for (size_t i = tid * elems_per_thd; i < num_elems_mid; i += elems_per_blk) {
  //       int idx_outer = i / num_elems_out_chunk;
  //       size_t idx_inner = i - idx_outer * num_elems_out_chunk;
  //       T* ns_data_val = (idx_outer == inter_dp_rank) ? (ns_data_out_self + idx_inner) :
  //       (ns_data_in + i); multimem_load_reduce(reg_data, mc_data_in + idx_outer * local_step +
  //       idx_inner); copy_data(ns_data_val, reg_data);
  //     }
  //     __syncthreads();

  //     // Gather data across different nodes
  //     size_t ns_msg_elems_out_chunk = floor_div<elems_per_thd>(num_elems_out_chunk);
  //     if (tid < num_nodes && tid != node_id) {
  //       int peer_world_rank = local_rank + tid * local_size;
  //       int peer_inter_dp_rank = tid / inter_tp_size;
  //       T const* src = (peer_inter_dp_rank == inter_dp_rank)
  //                          ? ns_data_out_self
  //                          : (ns_data_in + peer_inter_dp_rank * num_elems_out_chunk);
  //       nvshmem_put128_signal_nbi(ns_data_out_self, src, ns_msg_elems_out_chunk, ns_signal, 1,
  //                                 NVSHMEM_SIGNAL_ADD, peer_world_rank);
  //     }
  //     // Wait for gathered data to be ready
  //     if (tid == 0) {
  //       nvshmem_signal_wait_until(ns_signal, NVSHMEM_CMP_EQ, num_nodes - 1);
  //     }
  //     __syncthreads();

  //     if constexpr (local_tp_size > 1) {
  //       for (size_t i = tid * elems_per_thd; i < num_elems_out_chunk; i += elems_per_blk) {
  //         // Reduce data across different nodes
  //         signal_mode_reduce(reg_data, ns_data_out + i, num_elems_out_chunk, num_nodes);
  //         // Gather data in the same node
  //         multimem_store(mc_data_out + i, reg_data);
  //       }
  //       __syncthreads();
  //       // Wait for gathered data to be ready
  //       multimem_sync(mc_signal_out, mem_signal_out, local_size, tid);
  //       // Write outputs
  //       for (size_t i = tid * elems_per_thd; i < num_elems_out; i += elems_per_blk) {
  //         copy_data(x_out + i, mem_data_out + i);
  //       }
  //     } else {
  //       for (size_t i = tid * elems_per_thd; i < num_elems_out; i += elems_per_blk) {
  //         // Reduce data across different nodes
  //         signal_mode_reduce(reg_data, ns_data_out + i, num_elems_out_chunk, num_nodes);
  //         // Write outputs
  //         copy_data(x_out + i, reg_data);
  //       }
  //     }
  //     // Reset signals
  //     if (tid == 0) {
  //       if constexpr (local_tp_size > 1) {
  //         *reinterpret_cast<uint64_t*>(mem_signal_in) = 0;
  //       } else {
  //         *mem_signal_in = 0;
  //       }
  //       *ns_signal = 0;
  //     }
  //   }
}

}  // namespace mixed_comm
}  // namespace flashinfer
