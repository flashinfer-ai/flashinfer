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

constexpr int WARP_SIZE = 32;
constexpr int ACCESS_BYTES = 16;

enum MixedCommMode {
  OPT_WAITS,
  OPT_BYTES,
  MIXED,
};

__host__ __device__ constexpr bool is_power_of_2(int x) {
  return (x == (1 << 0) || x == (1 << 1) || x == (1 << 2) || x == (1 << 3) || x == (1 << 4) ||
          x == (1 << 5) || x == (1 << 6) || x == (1 << 7) || x == (1 << 8) || x == (1 << 9) ||
          x == (1 << 10) || x == (1 << 11) || x == (1 << 12) || x == (1 << 13) || x == (1 << 14) ||
          x == (1 << 15) || x == (1 << 16) || x == (1 << 17) || x == (1 << 18) || x == (1 << 19) ||
          x == (1 << 20) || x == (1 << 21) || x == (1 << 22) || x == (1 << 23) || x == (1 << 24) ||
          x == (1 << 25) || x == (1 << 26) || x == (1 << 27) || x == (1 << 28) || x == (1 << 29) ||
          x == (1 << 30) || x == (1 << 31));
}

template <int num_bits = 32, bool is_root = true>
__host__ __device__ constexpr int floor_log2_impl(int x) {
  static_assert(is_power_of_2(num_bits), "num_bits must be a power of 2");
  if constexpr (is_root) {
    return x == 0 ? -1 : floor_log2_impl<num_bits, false>(x);
  } else if constexpr (num_bits > 1) {
    constexpr int num_bits_next = num_bits / 2;
    constexpr int cmp_val = 1 << num_bits_next;
    return (x >= cmp_val)
               ? (num_bits_next | floor_log2_impl<num_bits_next, false>(x >> num_bits_next))
               : floor_log2_impl<num_bits_next, false>(x);
  } else {
    return 0;
  }
}

__host__ __device__ constexpr int floor_log2(int x) { return floor_log2_impl<32, true>(x); }

__host__ __device__ constexpr int ceil_log2(int x) {
  return floor_log2(x) + (is_power_of_2(x) ? 0 : 1);
}

template <int y, typename T>
__host__ __device__ constexpr T mod(T x) {
  if constexpr (is_power_of_2(y)) {
    constexpr T mask = y - 1;
    return x & mask;
  } else {
    return x % y;
  }
}

template <int y, typename T>
__host__ __device__ constexpr T floor_div(T x) {
  if constexpr (is_power_of_2(y)) {
    constexpr int shift_bits = floor_log2(y);
    return x >> shift_bits;
  } else {
    return x / y;
  }
}

template <typename T_x, typename T_y>
__host__ __device__ constexpr T_x ceil_div(T_x x, T_y y) {
  T_x res = y - 1;
  return (x + res) / y;
}

template <int y, typename T>
__host__ __device__ constexpr T ceil_div(T x) {
  constexpr T res = y - 1;
  return floor_div<y>(x + res);
}

template <typename T_x, typename T_y>
__host__ __device__ constexpr T_x round_down(T_x x, T_y y) {
  return (x / y) * y;
}

template <int y, typename T>
__host__ __device__ constexpr T round_down(T x) {
  if constexpr (is_power_of_2(y)) {
    constexpr T mask = y - 1;
    return x & ~mask;
  } else {
    return round_down(x, y);
  }
}

template <typename T_x, typename T_y>
__host__ __device__ constexpr T_x round_up(T_x x, T_y y) {
  T_x res = y - 1;
  return round_down(x + res, y);
}

template <int y, typename T>
__host__ __device__ constexpr T round_up(T x) {
  constexpr T res = y - 1;
  return round_down<y>(x + res);
}

template <int val>
inline __device__ void multimem_add_signal(uint32_t* signal) {
  asm volatile("multimem.red.release.sys.global.add.u32 [%0], %1;"
               :
               : "l"(signal), "n"(val)
               : "memory");
}

inline __device__ void wait_signal(uint32_t* signal, uint32_t val) {
  cuda::atomic_ref<uint32_t, cuda::thread_scope_system> ref_signal(*signal);
  while (ref_signal.load(cuda::memory_order_acquire) != val);
}

inline __device__ void multimem_sync(uint32_t* mc_signal, uint32_t* mem_signal, int size, int tid) {
  if (tid == 0) {
    multimem_add_signal<1>(mc_signal);
    cuda::ptx::fence_proxy_alias();
    wait_signal(mem_signal, size);
  }
  __syncthreads();
}

template <typename T>
inline __device__ void multimem_ld_reduce_vec(T* outputs, T const* inputs) {
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
inline __device__ void multimem_st_vec(T* outputs, T const* inputs) {
  float* dst = reinterpret_cast<float*>(outputs);
  float const* src = reinterpret_cast<float const*>(inputs);
  asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(dst), "f"(src[0]), "f"(src[1]), "f"(src[2]), "f"(src[3])
               : "memory");
}

template <int elems_per_thd, typename T_in, typename T_out>
inline __device__ void convert_vec(T_out* outputs, T_in const* inputs) {
  static_assert(!std::is_same_v<T_in, T_out>, "T_in and T_out must be different");
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

template <int elems_per_thd, typename T, typename T_ACC>
inline __device__ void accumulate_vec(T_ACC* outputs, T const* inputs) {
  if constexpr (std::is_same_v<T, float> && std::is_same_v<T_ACC, float>) {
#pragma unroll
    for (int i = 0; i < elems_per_thd; ++i) {
      outputs[i] += inputs[i];
    }
  } else {
    if constexpr (std::is_same_v<T_ACC, float>) {
      T_ACC reg_cvt[elems_per_thd];
      convert_vec<elems_per_thd>(reg_cvt, inputs);
      accumulate_vec<elems_per_thd>(outputs, reg_cvt);
    } else {
      static_assert(std::is_same_v<T, T_ACC>, "Unsupported data type");
#pragma unroll
      for (int i = 0; i < elems_per_thd; i += 2) {
        if constexpr (std::is_same_v<T, nv_half>) {
          *reinterpret_cast<nv_half2*>(outputs + i) +=
              *reinterpret_cast<nv_half2 const*>(inputs + i);
        } else {
          static_assert(std::is_same_v<T, nv_bfloat16>, "Unsupported data type");
          *reinterpret_cast<nv_bfloat162*>(outputs + i) +=
              *reinterpret_cast<nv_bfloat162 const*>(inputs + i);
        }
      }
    }
  }
}

#define REDUCE_DATA_PRE(reg_data, reg_acc, data, stride, elems_per_thd, T, T_ACC)          \
  do {                                                                                     \
    if constexpr (std::is_same_v<T, T_ACC>) {                                              \
      *reinterpret_cast<int4*>(reg_acc) = *reinterpret_cast<int4 const*>(data);            \
    } else {                                                                               \
      *reinterpret_cast<int4*>(reg_data[0]) = *reinterpret_cast<int4 const*>(data);        \
      convert_vec<elems_per_thd>(reg_acc, reg_data[0]);                                    \
    }                                                                                      \
    *reinterpret_cast<int4*>(reg_data[1]) = *reinterpret_cast<int4 const*>(data + stride); \
  } while (false)

#define REDUCE_DATA_MAIN(reg_data, reg_acc, data, stride, elems_per_thd, i) \
  do {                                                                      \
    int flag_reg_data = mod<2>(i - 1);                                      \
    *reinterpret_cast<int4*>(reg_data[flag_reg_data ^ 1]) =                 \
        *reinterpret_cast<int4 const*>(data + i * stride);                  \
    accumulate_vec<elems_per_thd>(reg_acc, reg_data[flag_reg_data]);        \
  } while (false)

#define REDUCE_DATA_POST(reg_data, reg_acc, reduce_size, elems_per_thd, T, T_ACC)    \
  do {                                                                               \
    if constexpr (std::is_same_v<T, T_ACC>) {                                        \
      accumulate_vec<elems_per_thd>(reg_data[0], reg_data[mod<2>(reduce_size - 1)]); \
    } else {                                                                         \
      accumulate_vec<elems_per_thd>(reg_acc, reg_data[mod<2>(reduce_size - 1)]);     \
      convert_vec<elems_per_thd>(reg_data[0], reg_acc);                              \
    }                                                                                \
  } while (false)

template <int reduce_size, int elems_per_thd, typename T, typename T_ACC>
inline __device__ void reduce_data(T reg_data[2][elems_per_thd], T_ACC reg_acc[elems_per_thd],
                                   T const* data, size_t stride) {
  REDUCE_DATA_PRE(reg_data, reg_acc, data, stride, elems_per_thd, T, T_ACC);
#pragma unroll
  for (int i = 2; i < reduce_size; ++i) {
    REDUCE_DATA_MAIN(reg_data, reg_acc, data, stride, elems_per_thd, i);
  }
  REDUCE_DATA_POST(reg_data, reg_acc, reduce_size, elems_per_thd, T, T_ACC);
}

template <int elems_per_thd, typename T, typename T_ACC>
inline __device__ void reduce_data(T reg_data[2][elems_per_thd], T_ACC reg_acc[elems_per_thd],
                                   T const* data, size_t stride, int reduce_size) {
  REDUCE_DATA_PRE(reg_data, reg_acc, data, stride, elems_per_thd, T, T_ACC);
  for (int i = 2; i < reduce_size; ++i) {
    REDUCE_DATA_MAIN(reg_data, reg_acc, data, stride, elems_per_thd, i);
  }
  REDUCE_DATA_POST(reg_data, reg_acc, reduce_size, elems_per_thd, T, T_ACC);
}

#undef REDUCE_DATA_PRE
#undef REDUCE_DATA_MAIN
#undef REDUCE_DATA_POST

template <int local_tp_size, int local_dp_size, int mode, typename T>
__global__ void fused_allreduce_allgather_single_node(
    T* x_out_all, T const* x_in_all, T* mem_data_all, uint32_t* mem_signal_all, T* mc_data_full_all,
    T* mc_data_tp_all, uint32_t* mc_signal_full_all, uint32_t* mc_signal_tp_all,
    size_t num_elems_in_all, int local_tp_rank, int local_dp_rank) {
  using T_ACC = float;
  constexpr int local_size = local_tp_size * local_dp_size;
  constexpr int elems_per_thd = ACCESS_BYTES / sizeof(T);
  int elems_per_blk = blockDim.x * elems_per_thd;
  int const& tid = threadIdx.x;
  int const& bid = blockIdx.x;

  // Divide inputs and outputs into blocks
  size_t max_num_elems_in;
  if constexpr (mode == MixedCommMode::OPT_WAITS) {
    max_num_elems_in = round_up<elems_per_thd>(ceil_div(num_elems_in_all, gridDim.x));
  } else {
    max_num_elems_in =
        round_up<elems_per_thd * local_tp_size>(ceil_div(num_elems_in_all, gridDim.x));
  }
  size_t ofst_elems_in = bid * max_num_elems_in;
  if (ofst_elems_in >= num_elems_in_all) {
    return;
  }
  size_t num_elems_in = min(num_elems_in_all - ofst_elems_in, max_num_elems_in);
  size_t num_elems_out = local_dp_size * num_elems_in;
  T* x_out = x_out_all + ofst_elems_in;
  T const* x_in = x_in_all + ofst_elems_in;

  if constexpr (mode == MixedCommMode::OPT_WAITS) {
    // Max virtual memory buffer size:
    //   max_local_bytes * local_size

    // Divide virtual memory into blocks
    size_t ofst_data = local_size * ofst_elems_in;
    T* mem_data = mem_data_all + ofst_data;
    T* mc_data = mc_data_full_all +
                 (ofst_data + (local_dp_rank + local_tp_rank * local_dp_size) * num_elems_in);
    uint32_t* mem_signal = mem_signal_all + bid;
    uint32_t* mc_signal = mc_signal_full_all + bid;
    T reg_data[2][elems_per_thd];
    T_ACC reg_acc[elems_per_thd];

    for (size_t i = tid * elems_per_thd; i < num_elems_in; i += elems_per_blk) {
      // Read inputs
      *reinterpret_cast<int4*>(reg_data[0]) = *reinterpret_cast<int4 const*>(x_in + i);
      // Gather data
      multimem_st_vec(mc_data + i, reg_data[0]);
    }
    __syncthreads();
    // Wait for gathered data to be ready
    multimem_sync(mc_signal, mem_signal, local_size, tid);

    for (size_t i = tid * elems_per_thd; i < num_elems_out; i += elems_per_blk) {
      int idx_outer = i / num_elems_in;
      size_t idx_inner = i - idx_outer * num_elems_in;
      T* x_out_val = x_out + (idx_outer * num_elems_in_all + idx_inner);
      if constexpr (local_tp_size > 1) {
        // Reduce data
        reduce_data<local_tp_size, elems_per_thd>(reg_data, reg_acc, mem_data + i, num_elems_out);
        // Write outputs
        *reinterpret_cast<int4*>(x_out_val) = *reinterpret_cast<int4 const*>(reg_data[0]);
      } else {
        // Write outputs
        *reinterpret_cast<int4*>(x_out_val) = *reinterpret_cast<int4 const*>(mem_data + i);
      }
    }
    // Reset signals
    if (tid == 0) {
      *mem_signal = 0;
    }
  } else {
    static_assert(mode == MixedCommMode::OPT_BYTES, "Unsupported mode");
    // Max virtual memory buffer size:
    //   max_local_bytes * min(local_size, local_dp_size + 1)

    // Divide virtual memory into blocks
    size_t num_elems_in_chunk = floor_div<local_tp_size>(max_num_elems_in);
    size_t ofst_elems_in_chunk = local_tp_rank * num_elems_in_chunk;
    T *mem_data_in, *mc_data_in, *mem_data_out, *mc_data_out;
    if constexpr (local_tp_size > 1) {
      size_t ofst_data_out = max_num_elems_in * gridDim.x + local_dp_size * ofst_elems_in;
      mem_data_in = mem_data_all + ofst_elems_in;
      mc_data_in = mc_data_tp_all + (ofst_elems_in + ofst_elems_in_chunk);
      mem_data_out = mem_data_all + ofst_data_out;
      mc_data_out = mc_data_full_all +
                    (ofst_data_out + local_dp_rank * max_num_elems_in + ofst_elems_in_chunk);
    } else {
      size_t ofst_data_out = local_dp_size * ofst_elems_in;
      mem_data_out = mem_data_all + ofst_data_out;
      mc_data_out = mc_data_full_all + (ofst_data_out + local_dp_rank * max_num_elems_in);
    }
    uint32_t *mem_signal_in, *mc_signal_in, *mem_signal_out, *mc_signal_out;
    if constexpr (local_tp_size > 1) {
      int bid_2 = bid * 2;
      mem_signal_in = mem_signal_all + bid_2;
      mc_signal_in = mc_signal_tp_all + bid_2;
      mem_signal_out = mem_signal_in + 1;
      mc_signal_out = mc_signal_full_all + (bid_2 + 1);
    } else {
      mem_signal_out = mem_signal_all + bid;
      mc_signal_out = mc_signal_full_all + bid;
    }
    T reg_data[elems_per_thd];

    if constexpr (local_tp_size > 1) {
      // Read inputs
      for (size_t i = tid * elems_per_thd; i < num_elems_in; i += elems_per_blk) {
        *reinterpret_cast<int4*>(mem_data_in + i) = *reinterpret_cast<int4 const*>(x_in + i);
      }
      __syncthreads();
      // Wait for inputs to be ready
      multimem_sync(mc_signal_in, mem_signal_in, local_tp_size, tid);

      for (size_t i = tid * elems_per_thd; i < num_elems_in_chunk; i += elems_per_blk) {
        // Reduce data
        multimem_ld_reduce_vec(reg_data, mc_data_in + i);
        // Gather data
        multimem_st_vec(mc_data_out + i, reg_data);
      }
    } else {
      for (size_t i = tid * elems_per_thd; i < num_elems_in; i += elems_per_blk) {
        // Read inputs
        *reinterpret_cast<int4*>(reg_data) = *reinterpret_cast<int4 const*>(x_in + i);
        // Gather data
        multimem_st_vec(mc_data_out + i, reg_data);
      }
    }
    __syncthreads();
    // Wait for gathered data to be ready
    multimem_sync(mc_signal_out, mem_signal_out, local_size, tid);
    // Write outputs
    for (size_t i = tid * elems_per_thd; i < num_elems_out; i += elems_per_blk) {
      int idx_outer = i / num_elems_in;
      size_t idx_inner = i - idx_outer * num_elems_in;
      *reinterpret_cast<int4*>(x_out + (idx_outer * num_elems_in_all + idx_inner)) =
          *reinterpret_cast<int4 const*>(mem_data_out + (idx_outer * max_num_elems_in + idx_inner));
    }
    // Reset signals
    if (tid == 0) {
      if constexpr (local_tp_size > 1) {
        *reinterpret_cast<uint64_t*>(mem_signal_in) = 0;
      } else {
        *mem_signal_out = 0;
      }
    }
  }
}

template <int local_tp_size, int local_dp_size, int mode, typename T>
__global__ void fused_allreduce_allgather_multi_node(
    T* x_out_all, T const* x_in_all, T* mem_data_all, uint32_t* mem_signal_all, T* mc_data_full_all,
    T* mc_data_tp_all, uint32_t* mc_signal_full_all, uint32_t* mc_signal_tp_all, T* ns_data_all,
    uint64_t* ns_signal_all, size_t num_elems_in_all, int local_tp_rank, int local_dp_rank,
    int inter_tp_rank, int inter_dp_rank, int inter_tp_size, int inter_dp_size) {
  using T_ACC = float;
  constexpr int local_size = local_tp_size * local_dp_size;
  int local_rank = local_tp_rank + local_dp_rank * local_tp_size;
  int num_nodes = inter_tp_size * inter_dp_size;
  int node_id = inter_tp_rank + inter_dp_rank * inter_tp_size;
  int dp_size = local_dp_size * inter_dp_size;
  constexpr int elems_per_thd = ACCESS_BYTES / sizeof(T);
  int elems_per_blk = blockDim.x * elems_per_thd;
  int const& tid = threadIdx.x;
  int const& bid = blockIdx.x;

  // Divide inputs and outputs into blocks
  size_t max_num_elems_in;
  if constexpr (mode == MixedCommMode::OPT_WAITS) {
    max_num_elems_in = round_up<elems_per_thd>(ceil_div(num_elems_in_all, gridDim.x));
  } else if constexpr (mode == MixedCommMode::OPT_BYTES) {
    max_num_elems_in = round_up(ceil_div(num_elems_in_all, gridDim.x),
                                elems_per_thd * local_tp_size * inter_tp_size);
  } else {
    max_num_elems_in =
        round_up<elems_per_thd * local_tp_size>(ceil_div(num_elems_in_all, gridDim.x));
  }
  size_t ofst_elems_in = bid * max_num_elems_in;
  if (ofst_elems_in >= num_elems_in_all) {
    return;
  }
  size_t num_elems_in = min(num_elems_in_all - ofst_elems_in, max_num_elems_in);
  size_t num_elems_out = dp_size * num_elems_in;
  T* x_out = x_out_all + ofst_elems_in;
  T const* x_in = x_in_all + ofst_elems_in;
  T reg_data[2][elems_per_thd];
  T_ACC reg_acc[elems_per_thd];

  if constexpr (mode == MixedCommMode::OPT_WAITS) {
    // Max virtual memory buffer size:
    //   max_local_bytes * inter_dp_size * local_size
    // Max nvshmem buffer size:
    //   max_local_bytes * num_nodes

    // Divide virtual memory and nvshmem buffers into blocks
    size_t num_elems_mid = inter_dp_size * num_elems_in;
    size_t ofst_data_local = local_size * inter_dp_size * ofst_elems_in;
    T* mem_data = mem_data_all + ofst_data_local;
    T* mc_data =
        mc_data_full_all +
        (ofst_data_local + (local_dp_rank + local_tp_rank * local_dp_size) * num_elems_mid);
    T* ns_data = ns_data_all + num_nodes * ofst_elems_in;
    T* ns_data_self = ns_data + (inter_dp_rank + inter_tp_rank * inter_dp_size) * num_elems_in;
    uint32_t* mem_signal = mem_signal_all + bid;
    uint32_t* mc_signal = mc_signal_full_all + bid;
    uint64_t* ns_signal = ns_signal_all + bid;

    // Read inputs
    for (size_t i = tid * elems_per_thd; i < num_elems_in; i += elems_per_blk) {
      *reinterpret_cast<int4*>(ns_data_self + i) = *reinterpret_cast<int4 const*>(x_in + i);
    }
    __syncthreads();
    // Gather data across different nodes
    size_t ns_msg_elems_in = floor_div<elems_per_thd>(num_elems_in);
    if (tid < num_nodes && tid != node_id) {
      int peer_world_rank = local_rank + tid * local_size;
      nvshmem_put128_signal_nbi(ns_data_self, ns_data_self, ns_msg_elems_in, ns_signal, 1,
                                NVSHMEM_SIGNAL_ADD, peer_world_rank);
    }
    // Wait for gathered data to be ready
    if (tid == 0) {
      nvshmem_signal_wait_until(ns_signal, NVSHMEM_CMP_EQ, num_nodes - 1);
    }
    __syncthreads();

    if (inter_tp_size > 1) {
      for (size_t i = tid * elems_per_thd; i < num_elems_mid; i += elems_per_blk) {
        // Reduce data across different nodes
        reduce_data<elems_per_thd>(reg_data, reg_acc, ns_data + i, num_elems_mid, inter_tp_size);
        // Gather data in the same node
        multimem_st_vec(mc_data + i, reg_data[0]);
      }
    } else {
      // Gather data in the same node
      for (size_t i = tid * elems_per_thd; i < num_elems_mid; i += elems_per_blk) {
        *reinterpret_cast<int4*>(reg_data[0]) = *reinterpret_cast<int4 const*>(ns_data + i);
        multimem_st_vec(mc_data + i, reg_data[0]);
      }
    }
    __syncthreads();
    // Wait for gathered data to be ready
    multimem_sync(mc_signal, mem_signal, local_size, tid);

    for (size_t i = tid * elems_per_thd; i < num_elems_out; i += elems_per_blk) {
      int idx_outer_raw = i / num_elems_in;
      int idx_outer_local = idx_outer_raw / inter_dp_size;
      int idx_outer_inter = idx_outer_raw - idx_outer_local * inter_dp_size;
      int idx_outer = idx_outer_local + idx_outer_inter * local_dp_size;
      size_t idx_inner = i - idx_outer_raw * num_elems_in;
      T* x_out_val = x_out + (idx_outer * num_elems_in_all + idx_inner);
      if constexpr (local_tp_size > 1) {
        // Reduce data in the same node
        reduce_data<local_tp_size, elems_per_thd>(reg_data, reg_acc, mem_data + i, num_elems_out);
        // Write outputs
        *reinterpret_cast<int4*>(x_out_val) = *reinterpret_cast<int4 const*>(reg_data[0]);
      } else {
        // Write outputs
        *reinterpret_cast<int4*>(x_out_val) = *reinterpret_cast<int4 const*>(mem_data + i);
      }
    }
    // Reset signals
    if (tid == 0) {
      *mem_signal = 0;
      *ns_signal = 0;
    }
  } else if constexpr (mode == MixedCommMode::OPT_BYTES) {
    // Max virtual memory buffer size:
    //   max_local_bytes * min(inter_dp_size * local_size, dp_size + 1)
    // Max nvshmem buffer size:
    //   max_local_bytes / local_tp_size * (inter_dp_size + 2)

    // Divide virtual memory and nvshmem buffers into blocks
    size_t num_elems_in_chunk = floor_div<local_tp_size>(max_num_elems_in);
    size_t ofst_elems_in_chunk = local_tp_rank * num_elems_in_chunk;
    size_t num_elems_mid_chunk = num_elems_in_chunk / inter_tp_size;
    size_t ofst_elems_mid_chunk = inter_tp_rank * num_elems_mid_chunk;
    size_t num_elems_mid = inter_dp_size * num_elems_in_chunk;
    T *mem_data_in, *mc_data_in, *mem_data_out, *mc_data_out;
    if constexpr (local_tp_size > 1) {
      size_t ofst_data_local_out = max_num_elems_in * gridDim.x + dp_size * ofst_elems_in;
      mem_data_in = mem_data_all + ofst_elems_in;
      mc_data_in = mc_data_tp_all + (ofst_elems_in + ofst_elems_in_chunk);
      mem_data_out = mem_data_all + ofst_data_local_out;
      mc_data_out = mc_data_full_all +
                    (ofst_data_local_out + local_dp_rank * max_num_elems_in + ofst_elems_in_chunk);
    } else {
      size_t ofst_data_local_out = dp_size * ofst_elems_in;
      mem_data_out = mem_data_all + ofst_data_local_out;
      mc_data_out = mc_data_full_all + (ofst_data_local_out + local_dp_rank * max_num_elems_in);
    }
    int bid_2 = bid * 2;
    uint32_t *mem_signal_in, *mc_signal_in, *mem_signal_out, *mc_signal_out;
    if constexpr (local_tp_size > 1) {
      mem_signal_in = mem_signal_all + bid_2;
      mc_signal_in = mc_signal_tp_all + bid_2;
      mem_signal_out = mem_signal_in + 1;
      mc_signal_out = mc_signal_full_all + (bid_2 + 1);
    } else {
      mem_signal_out = mem_signal_all + bid;
      mc_signal_out = mc_signal_full_all + bid;
    }
    T *ns_data_in, *ns_data_mid, *ns_data_out, *ns_data_out_self;
    uint64_t *ns_signal_in, *ns_signal_out;
    if (inter_tp_size > 1) {
      ns_data_in = ns_data_all + bid * (inter_dp_size + 2) * num_elems_in_chunk;
      ns_data_mid = ns_data_in + num_elems_in_chunk;
      ns_data_out = ns_data_mid + num_elems_in_chunk;
      ns_data_out_self = ns_data_out + node_id * num_elems_mid_chunk;
      ns_signal_in = ns_signal_all + bid_2;
      ns_signal_out = ns_signal_in + 1;
    } else {
      ns_data_out = ns_data_all + bid * num_elems_mid;
      ns_data_out_self = ns_data_out + inter_dp_rank * num_elems_in_chunk;
      ns_signal_out = ns_signal_all + bid;
    }

    if constexpr (local_tp_size > 1) {
      // Read inputs
      for (size_t i = tid * elems_per_thd; i < num_elems_in; i += elems_per_blk) {
        *reinterpret_cast<int4*>(mem_data_in + i) = *reinterpret_cast<int4 const*>(x_in + i);
      }
      __syncthreads();
      // Wait for inputs to be ready
      multimem_sync(mc_signal_in, mem_signal_in, local_tp_size, tid);
      // Reduce data in the same node
      if (inter_tp_size > 1) {
        for (size_t i = tid * elems_per_thd; i < num_elems_in_chunk; i += elems_per_blk) {
          int peer_inter_tp_rank = i / num_elems_mid_chunk;
          T* dst = (peer_inter_tp_rank == inter_tp_rank) ? ns_data_mid : ns_data_in;
          multimem_ld_reduce_vec(reg_data[0], mc_data_in + i);
          *reinterpret_cast<int4*>(dst + i) = *reinterpret_cast<int4 const*>(reg_data[0]);
        }
      } else {
        for (size_t i = tid * elems_per_thd; i < num_elems_in_chunk; i += elems_per_blk) {
          multimem_ld_reduce_vec(reg_data[0], mc_data_in + i);
          *reinterpret_cast<int4*>(ns_data_out_self + i) =
              *reinterpret_cast<int4 const*>(reg_data[0]);
        }
      }
    } else {
      // Read inputs
      if (inter_tp_size > 1) {
        for (size_t i = tid * elems_per_thd; i < num_elems_in; i += elems_per_blk) {
          int peer_inter_tp_rank = i / num_elems_mid_chunk;
          T* dst = (peer_inter_tp_rank == inter_tp_rank) ? ns_data_mid : ns_data_in;
          *reinterpret_cast<int4*>(dst + i) = *reinterpret_cast<int4 const*>(x_in + i);
        }
      } else {
        for (size_t i = tid * elems_per_thd; i < num_elems_in; i += elems_per_blk) {
          *reinterpret_cast<int4*>(ns_data_out_self + i) = *reinterpret_cast<int4 const*>(x_in + i);
        }
      }
    }
    __syncthreads();

    size_t ns_msg_elems_mid_chunk = floor_div<elems_per_thd>(num_elems_mid_chunk);
    if (inter_tp_size > 1) {
      // Gather data across different nodes
      if (tid < inter_tp_size && tid != inter_tp_rank) {
        int peer_world_rank = local_rank + (tid + inter_dp_rank * inter_tp_size) * local_size;
        T* dst = ns_data_mid + inter_tp_rank * num_elems_mid_chunk;
        T const* src = ns_data_in + tid * num_elems_mid_chunk;
        nvshmem_put128_signal_nbi(dst, src, ns_msg_elems_mid_chunk, ns_signal_in, 1,
                                  NVSHMEM_SIGNAL_ADD, peer_world_rank);
      }
      // Wait for gathered data to be ready
      if (tid == 0) {
        nvshmem_signal_wait_until(ns_signal_in, NVSHMEM_CMP_EQ, inter_tp_size - 1);
      }
      __syncthreads();
      // Reduce data across different nodes
      for (size_t i = tid * elems_per_thd; i < num_elems_mid_chunk; i += elems_per_blk) {
        reduce_data<elems_per_thd>(reg_data, reg_acc, ns_data_mid + i, num_elems_mid_chunk,
                                   inter_tp_size);
        *reinterpret_cast<int4*>(ns_data_out_self + i) =
            *reinterpret_cast<int4 const*>(reg_data[0]);
      }
      __syncthreads();
    }
    // Gather data across different nodes
    if (tid < num_nodes && tid != node_id) {
      int peer_world_rank = local_rank + tid * local_size;
      nvshmem_put128_signal_nbi(ns_data_out_self, ns_data_out_self, ns_msg_elems_mid_chunk,
                                ns_signal_out, 1, NVSHMEM_SIGNAL_ADD, peer_world_rank);
    }
    // Wait for gathered data to be ready
    if (tid == 0) {
      nvshmem_signal_wait_until(ns_signal_out, NVSHMEM_CMP_EQ, num_nodes - 1);
    }
    __syncthreads();

    // Gather data in the same node
    size_t local_step = local_dp_size * max_num_elems_in;
    for (size_t i = tid * elems_per_thd; i < num_elems_mid; i += elems_per_blk) {
      int idx_outer = i / num_elems_in_chunk;
      size_t idx_inner = i - idx_outer * num_elems_in_chunk;
      *reinterpret_cast<int4*>(reg_data[0]) = *reinterpret_cast<int4 const*>(ns_data_out + i);
      multimem_st_vec(mc_data_out + (idx_outer * local_step + idx_inner), reg_data[0]);
    }
    __syncthreads();
    // Wait for gathered data to be ready
    multimem_sync(mc_signal_out, mem_signal_out, local_size, tid);
    // Write outputs
    for (size_t i = tid * elems_per_thd; i < num_elems_out; i += elems_per_blk) {
      int idx_outer = i / num_elems_in;
      size_t idx_inner = i - idx_outer * num_elems_in;
      *reinterpret_cast<int4*>(x_out + (idx_outer * num_elems_in_all + idx_inner)) =
          *reinterpret_cast<int4 const*>(mem_data_out + (idx_outer * max_num_elems_in + idx_inner));
    }
    // Reset signals
    if (tid == 0) {
      if constexpr (local_tp_size > 1) {
        *reinterpret_cast<uint64_t*>(mem_signal_in) = 0;
      } else {
        *mem_signal_out = 0;
      }
      if (inter_tp_size > 1) {
        *reinterpret_cast<__uint128_t*>(ns_signal_in) = 0;
      } else {
        *ns_signal_out = 0;
      }
    }
  } else {
    static_assert(mode == MixedCommMode::MIXED, "Unsupported mode");
    // Max virtual memory buffer size:
    //   max_local_bytes * min(inter_dp_size * local_size, dp_size + 1)
    // Max nvshmem buffer size:
    //   max_local_bytes / local_tp_size * num_nodes

    // Divide virtual memory and nvshmem buffers into blocks
    size_t num_elems_in_chunk = floor_div<local_tp_size>(max_num_elems_in);
    size_t ofst_elems_in_chunk = local_tp_rank * num_elems_in_chunk;
    size_t num_elems_mid = inter_dp_size * num_elems_in_chunk;
    T *mem_data_in, *mc_data_in, *mem_data_out, *mc_data_out;
    if constexpr (local_tp_size > 1) {
      size_t ofst_data_local_out = max_num_elems_in * gridDim.x + dp_size * ofst_elems_in;
      mem_data_in = mem_data_all + ofst_elems_in;
      mc_data_in = mc_data_tp_all + (ofst_elems_in + ofst_elems_in_chunk);
      mem_data_out = mem_data_all + ofst_data_local_out;
      mc_data_out = mc_data_full_all +
                    (ofst_data_local_out + local_dp_rank * max_num_elems_in + ofst_elems_in_chunk);
    } else {
      size_t ofst_data_local_out = dp_size * ofst_elems_in;
      mem_data_out = mem_data_all + ofst_data_local_out;
      mc_data_out = mc_data_full_all + (ofst_data_local_out + local_dp_rank * max_num_elems_in);
    }
    T* ns_data = ns_data_all + bid * num_nodes * num_elems_in_chunk;
    T* ns_data_self =
        ns_data + (inter_dp_rank + inter_tp_rank * inter_dp_size) * num_elems_in_chunk;
    uint32_t *mem_signal_in, *mc_signal_in, *mem_signal_out, *mc_signal_out;
    if constexpr (local_tp_size > 1) {
      int bid_2 = bid * 2;
      mem_signal_in = mem_signal_all + bid_2;
      mc_signal_in = mc_signal_tp_all + bid_2;
      mem_signal_out = mem_signal_in + 1;
      mc_signal_out = mc_signal_full_all + (bid_2 + 1);
    } else {
      mem_signal_out = mem_signal_all + bid;
      mc_signal_out = mc_signal_full_all + bid;
    }
    uint64_t* ns_signal = ns_signal_all + bid;

    if constexpr (local_tp_size > 1) {
      // Read inputs
      for (size_t i = tid * elems_per_thd; i < num_elems_in; i += elems_per_blk) {
        *reinterpret_cast<int4*>(mem_data_in + i) = *reinterpret_cast<int4 const*>(x_in + i);
      }
      __syncthreads();
      // Wait for inputs to be ready
      multimem_sync(mc_signal_in, mem_signal_in, local_tp_size, tid);
      // Reduce data in the same node
      for (size_t i = tid * elems_per_thd; i < num_elems_in_chunk; i += elems_per_blk) {
        multimem_ld_reduce_vec(reg_data[0], mc_data_in + i);
        *reinterpret_cast<int4*>(ns_data_self + i) = *reinterpret_cast<int4 const*>(reg_data[0]);
      }
    } else {
      // Read inputs
      for (size_t i = tid * elems_per_thd; i < num_elems_in; i += elems_per_blk) {
        *reinterpret_cast<int4*>(ns_data_self + i) = *reinterpret_cast<int4 const*>(x_in + i);
      }
    }
    __syncthreads();

    // Gather data across different nodes
    size_t ns_msg_elems_in_chunk = floor_div<elems_per_thd>(num_elems_in_chunk);
    if (tid < num_nodes && tid != node_id) {
      int peer_world_rank = local_rank + tid * local_size;
      nvshmem_put128_signal_nbi(ns_data_self, ns_data_self, ns_msg_elems_in_chunk, ns_signal, 1,
                                NVSHMEM_SIGNAL_ADD, peer_world_rank);
    }
    // Wait for gathered data to be ready
    if (tid == 0) {
      nvshmem_signal_wait_until(ns_signal, NVSHMEM_CMP_EQ, num_nodes - 1);
    }
    __syncthreads();

    size_t local_step = local_dp_size * max_num_elems_in;
    if (inter_tp_size > 1) {
      for (size_t i = tid * elems_per_thd; i < num_elems_mid; i += elems_per_blk) {
        int idx_outer = i / num_elems_in_chunk;
        size_t idx_inner = i - idx_outer * num_elems_in_chunk;
        // Reduce data across different nodes
        reduce_data<elems_per_thd>(reg_data, reg_acc, ns_data + i, num_elems_mid, inter_tp_size);
        // Gather data in the same node
        multimem_st_vec(mc_data_out + (idx_outer * local_step + idx_inner), reg_data[0]);
      }
    } else {
      // Gather data in the same node
      for (size_t i = tid * elems_per_thd; i < num_elems_mid; i += elems_per_blk) {
        int idx_outer = i / num_elems_in_chunk;
        size_t idx_inner = i - idx_outer * num_elems_in_chunk;
        *reinterpret_cast<int4*>(reg_data[0]) = *reinterpret_cast<int4 const*>(ns_data + i);
        multimem_st_vec(mc_data_out + (idx_outer * local_step + idx_inner), reg_data[0]);
      }
    }
    __syncthreads();
    // Wait for gathered data to be ready
    multimem_sync(mc_signal_out, mem_signal_out, local_size, tid);
    // Write outputs
    for (size_t i = tid * elems_per_thd; i < num_elems_out; i += elems_per_blk) {
      int idx_outer = i / num_elems_in;
      size_t idx_inner = i - idx_outer * num_elems_in;
      *reinterpret_cast<int4*>(x_out + (idx_outer * num_elems_in_all + idx_inner)) =
          *reinterpret_cast<int4 const*>(mem_data_out + (idx_outer * max_num_elems_in + idx_inner));
    }
    // Reset signals
    if (tid == 0) {
      if constexpr (local_tp_size > 1) {
        *reinterpret_cast<uint64_t*>(mem_signal_in) = 0;
      } else {
        *mem_signal_out = 0;
      }
      *ns_signal = 0;
    }
  }
}

template <int local_tp_size, int local_dp_size, int mode, typename T>
__global__ void fused_reducescatter_allreduce_single_node(
    T* x_out_all, T const* x_in_all, T* mem_data_all, uint32_t* mem_signal_all,
    T* const* uc_data_all_array, T* mc_data_full_all, T* mc_data_tp_all,
    uint32_t* mc_signal_full_all, size_t num_elems_out_all, int local_tp_rank, int local_dp_rank) {
  using T_ACC = float;
  constexpr int local_size = local_tp_size * local_dp_size;
  constexpr int elems_per_thd = ACCESS_BYTES / sizeof(T);
  int elems_per_blk = blockDim.x * elems_per_thd;
  int const& tid = threadIdx.x;
  int const& bid = blockIdx.x;

  // Divide inputs and outputs into blocks
  size_t max_num_elems_out;
  if constexpr (mode == MixedCommMode::OPT_WAITS) {
    max_num_elems_out = round_up<elems_per_thd>(ceil_div(num_elems_out_all, gridDim.x));
  } else {
    max_num_elems_out =
        round_up<elems_per_thd * local_tp_size>(ceil_div(num_elems_out_all, gridDim.x));
  }
  size_t ofst_elems_out = bid * max_num_elems_out;
  if (ofst_elems_out >= num_elems_out_all) {
    return;
  }
  size_t num_elems_out = min(num_elems_out_all - ofst_elems_out, max_num_elems_out);
  size_t num_elems_in = local_dp_size * num_elems_out;
  T* x_out = x_out_all + ofst_elems_out;
  T const* x_in = x_in_all + ofst_elems_out;

  if constexpr (mode == MixedCommMode::OPT_WAITS) {
    // Max virtual memory buffer size:
    //   max_local_bytes * local_size

    // Divide virtual memory into blocks
    size_t ofst_data = local_size * ofst_elems_out;
    size_t ofst_data_self =
        ofst_data + (local_tp_rank + local_dp_rank * local_tp_size) * num_elems_out;
    T* mem_data = mem_data_all + ofst_data;
    T* mc_data = mc_data_full_all + ofst_data_self;
    T* uc_data_array[local_size];
    if constexpr (local_tp_size > 1) {
#pragma unroll
      for (int i = 0; i < local_size; ++i) {
        uc_data_array[i] = uc_data_all_array[i] + ofst_data_self;
      }
    }
    uint32_t* mem_signal = mem_signal_all + bid;
    uint32_t* mc_signal = mc_signal_full_all + bid;
    T reg_data[2][elems_per_thd];
    T_ACC reg_acc[elems_per_thd];

    for (size_t i = tid * elems_per_thd; i < num_elems_in; i += elems_per_blk) {
      int idx_outer = i / num_elems_out;
      size_t idx_inner = i - idx_outer * num_elems_out;
      T const* x_in_val = x_in + (idx_outer * num_elems_out_all + idx_inner);
      if constexpr (local_tp_size > 1) {
        // Read inputs
        *reinterpret_cast<int4*>(reg_data[0]) = *reinterpret_cast<int4 const*>(x_in_val);
        // Gather data
        int peer_local_rank_base = idx_outer * local_tp_size;
#pragma unroll
        for (int j = 0; j < local_tp_size; ++j) {
          int peer_local_rank = peer_local_rank_base + mod<local_tp_size>(local_tp_rank + j);
          *reinterpret_cast<int4*>(uc_data_array[peer_local_rank] + idx_inner) =
              *reinterpret_cast<int4 const*>(reg_data[0]);
        }
      } else {
        // Read inputs
        *reinterpret_cast<int4*>(mem_data + i) = *reinterpret_cast<int4 const*>(x_in_val);
      }
    }
    __syncthreads();
    // Wait for inputs or gathered data to be ready
    multimem_sync(mc_signal, mem_signal, local_size, tid);

    for (size_t i = tid * elems_per_thd; i < num_elems_out; i += elems_per_blk) {
      // Reduce data
      if constexpr (local_tp_size > 1) {
        reduce_data<local_size, elems_per_thd>(reg_data, reg_acc, mem_data + i, num_elems_out);
      } else {
        multimem_ld_reduce_vec(reg_data[0], mc_data + i);
      }
      // Write outputs
      *reinterpret_cast<int4*>(x_out + i) = *reinterpret_cast<int4 const*>(reg_data[0]);
    }
    // Reset signals
    if (tid == 0) {
      *mem_signal = 0;
    }
  } else {
    static_assert(mode == MixedCommMode::OPT_BYTES, "Unsupported mode");
    // Max virtual memory buffer size:
    //   max_local_bytes * min(local_size, local_dp_size + 1)

    // Divide virtual memory into blocks
    size_t num_elems_out_chunk = floor_div<local_tp_size>(max_num_elems_out);
    size_t ofst_elems_out_chunk = local_tp_rank * num_elems_out_chunk;
    T *mem_data_in, *mc_data_in, *mem_data_out, *mc_data_out;
    size_t ofst_data_in = local_dp_size * ofst_elems_out;
    if constexpr (local_tp_size > 1) {
      size_t ofst_data_out = local_dp_size * max_num_elems_out * gridDim.x + ofst_elems_out;
      mem_data_in = mem_data_all + ofst_data_in;
      mc_data_in = mc_data_full_all +
                   (ofst_data_in + local_dp_rank * max_num_elems_out + ofst_elems_out_chunk);
      mem_data_out = mem_data_all + ofst_data_out;
      mc_data_out = mc_data_tp_all + (ofst_data_out + ofst_elems_out_chunk);
    } else {
      mem_data_in = mem_data_all + ofst_data_in;
      mc_data_in = mc_data_full_all + (ofst_data_in + local_dp_rank * max_num_elems_out);
    }
    uint32_t *mem_signal_in, *mc_signal_in, *mem_signal_out, *mc_signal_out;
    if constexpr (local_tp_size > 1) {
      int bid_2 = bid * 2;
      mem_signal_in = mem_signal_all + bid_2;
      mc_signal_in = mc_signal_full_all + bid_2;
      mem_signal_out = mem_signal_in + 1;
      mc_signal_out = mc_signal_in + 1;
    } else {
      mem_signal_in = mem_signal_all + bid;
      mc_signal_in = mc_signal_full_all + bid;
    }
    T reg_data[elems_per_thd];

    // Read inputs
    for (size_t i = tid * elems_per_thd; i < num_elems_in; i += elems_per_blk) {
      int idx_outer = i / num_elems_out;
      size_t idx_inner = i - idx_outer * num_elems_out;
      *reinterpret_cast<int4*>(mem_data_in + (idx_outer * max_num_elems_out + idx_inner)) =
          *reinterpret_cast<int4 const*>(x_in + (idx_outer * num_elems_out_all + idx_inner));
    }
    __syncthreads();
    // Wait for inputs to be ready
    multimem_sync(mc_signal_in, mem_signal_in, local_size, tid);

    if constexpr (local_tp_size > 1) {
      for (size_t i = tid * elems_per_thd; i < num_elems_out_chunk; i += elems_per_blk) {
        // Reduce data
        multimem_ld_reduce_vec(reg_data, mc_data_in + i);
        // Gather data
        multimem_st_vec(mc_data_out + i, reg_data);
      }
      __syncthreads();
      // Wait for gathered data to be ready
      multimem_sync(mc_signal_out, mem_signal_out, local_size, tid);
      // Write outputs
      for (size_t i = tid * elems_per_thd; i < num_elems_out; i += elems_per_blk) {
        *reinterpret_cast<int4*>(x_out + i) = *reinterpret_cast<int4 const*>(mem_data_out + i);
      }
    } else {
      for (size_t i = tid * elems_per_thd; i < num_elems_out; i += elems_per_blk) {
        // Reduce data
        multimem_ld_reduce_vec(reg_data, mc_data_in + i);
        // Write outputs
        *reinterpret_cast<int4*>(x_out + i) = *reinterpret_cast<int4 const*>(reg_data);
      }
    }
    // Reset signals
    if (tid == 0) {
      if constexpr (local_tp_size > 1) {
        *reinterpret_cast<uint64_t*>(mem_signal_in) = 0;
      } else {
        *mem_signal_in = 0;
      }
    }
  }
}

template <int local_tp_size, int local_dp_size, int mode, typename T>
__global__ void fused_reducescatter_allreduce_multi_node(
    T* x_out_all, T const* x_in_all, T* mem_data_all, uint32_t* mem_signal_all,
    T* const* uc_data_all_array, T* mc_data_full_all, T* mc_data_tp_all,
    uint32_t* mc_signal_full_all, T* ns_data_all, uint64_t* ns_signal_all, size_t num_elems_out_all,
    int local_tp_rank, int local_dp_rank, int inter_tp_rank, int inter_dp_rank, int inter_tp_size,
    int inter_dp_size) {
  using T_ACC = float;
  constexpr int local_size = local_tp_size * local_dp_size;
  int local_rank = local_tp_rank + local_dp_rank * local_tp_size;
  int num_nodes = inter_tp_size * inter_dp_size;
  int node_id = inter_tp_rank + inter_dp_rank * inter_tp_size;
  int dp_size = local_dp_size * inter_dp_size;
  constexpr int elems_per_thd = ACCESS_BYTES / sizeof(T);
  int elems_per_blk = blockDim.x * elems_per_thd;
  int const& tid = threadIdx.x;
  int const& bid = blockIdx.x;

  // Divide inputs and outputs into blocks
  size_t max_num_elems_out;
  if constexpr (mode == MixedCommMode::OPT_WAITS) {
    max_num_elems_out = round_up<elems_per_thd>(ceil_div(num_elems_out_all, gridDim.x));
  } else if constexpr (mode == MixedCommMode::OPT_BYTES) {
    max_num_elems_out = round_up(ceil_div(num_elems_out_all, gridDim.x),
                                 elems_per_thd * local_tp_size * inter_tp_size);
  } else {
    max_num_elems_out =
        round_up<elems_per_thd * local_tp_size>(ceil_div(num_elems_out_all, gridDim.x));
  }
  size_t ofst_elems_out = bid * max_num_elems_out;
  if (ofst_elems_out >= num_elems_out_all) {
    return;
  }
  size_t num_elems_out = min(num_elems_out_all - ofst_elems_out, max_num_elems_out);
  size_t num_elems_in = dp_size * num_elems_out;
  T* x_out = x_out_all + ofst_elems_out;
  T const* x_in = x_in_all + ofst_elems_out;
  T reg_data[2][elems_per_thd];
  T_ACC reg_acc[elems_per_thd];

  if constexpr (mode == MixedCommMode::OPT_WAITS) {
    // Max virtual memory buffer size:
    //   max_local_bytes * inter_dp_size * local_size
    // Max nvshmem buffer size:
    //   max_local_bytes * (inter_dp_size + num_nodes)

    // Divide virtual memory and nvshmem buffers into blocks
    size_t num_elems_mid = inter_dp_size * num_elems_out;
    size_t ofst_data_local = local_size * inter_dp_size * ofst_elems_out;
    size_t ofst_data_local_self = ofst_data_local + local_rank * num_elems_mid;
    T* mem_data = mem_data_all + ofst_data_local;
    T* mc_data = mc_data_full_all + ofst_data_local_self;
    T* uc_data_array[local_size];
    if constexpr (local_tp_size > 1) {
#pragma unroll
      for (int i = 0; i < local_size; ++i) {
        uc_data_array[i] = uc_data_all_array[i] + ofst_data_local_self;
      }
    }
    T* ns_data_in = ns_data_all + (inter_dp_size + num_nodes) * ofst_elems_out;
    T* ns_data_out = ns_data_in + inter_dp_size * num_elems_out;
    T* ns_data_out_self = ns_data_out + node_id * num_elems_out;
    uint32_t* mem_signal = mem_signal_all + bid;
    uint32_t* mc_signal = mc_signal_full_all + bid;
    uint64_t* ns_signal = ns_signal_all + bid;

    for (size_t i = tid * elems_per_thd; i < num_elems_in; i += elems_per_blk) {
      int idx_outer_in = i / num_elems_out;
      size_t idx_inner = i - idx_outer_in * num_elems_out;
      T const* x_in_val = x_in + (idx_outer_in * num_elems_out_all + idx_inner);
      int idx_outer_out_inter = idx_outer_in / local_dp_size;
      int idx_outer_out_local = idx_outer_in - idx_outer_out_inter * local_dp_size;
      if constexpr (local_tp_size > 1) {
        // Read inputs
        *reinterpret_cast<int4*>(reg_data[0]) = *reinterpret_cast<int4 const*>(x_in_val);
        // Gather data
        int peer_local_rank_base = idx_outer_out_local * local_tp_size;
        size_t ofst_uc_data = idx_outer_out_inter * num_elems_out + idx_inner;
#pragma unroll
        for (int j = 0; j < local_tp_size; ++j) {
          int peer_local_rank = peer_local_rank_base + mod<local_tp_size>(local_tp_rank + j);
          *reinterpret_cast<int4*>(uc_data_array[peer_local_rank] + ofst_uc_data) =
              *reinterpret_cast<int4 const*>(reg_data[0]);
        }
      } else {
        // Read inputs
        int idx_outer_out = idx_outer_out_inter + idx_outer_out_local * inter_dp_size;
        *reinterpret_cast<int4*>(mem_data + (idx_outer_out * num_elems_out + idx_inner)) =
            *reinterpret_cast<int4 const*>(x_in_val);
      }
    }
    __syncthreads();
    // Wait for inputs or gathered data to be ready
    multimem_sync(mc_signal, mem_signal, local_size, tid);
    // Reduce data in the same node
    for (size_t i = tid * elems_per_thd; i < num_elems_mid; i += elems_per_blk) {
      int idx_outer = i / num_elems_out;
      size_t idx_inner = i - idx_outer * num_elems_out;
      T* ns_data_val = (idx_outer == inter_dp_rank) ? ns_data_out_self + idx_inner : ns_data_in + i;
      if constexpr (local_tp_size > 1) {
        reduce_data<local_size, elems_per_thd>(reg_data, reg_acc, mem_data + i, num_elems_mid);
      } else {
        multimem_ld_reduce_vec(reg_data[0], mc_data + i);
      }
      *reinterpret_cast<int4*>(ns_data_val) = *reinterpret_cast<int4 const*>(reg_data[0]);
    }
    __syncthreads();
    // Gather data across different nodes
    size_t ns_msg_elems_out = floor_div<elems_per_thd>(num_elems_out);
    if (tid < num_nodes && tid != node_id) {
      int peer_world_rank = local_rank + tid * local_size;
      int peer_inter_dp_rank = tid / inter_tp_size;
      T const* src = (peer_inter_dp_rank == inter_dp_rank)
                         ? ns_data_out_self
                         : ns_data_in + peer_inter_dp_rank * num_elems_out;
      nvshmem_put128_signal_nbi(ns_data_out_self, src, ns_msg_elems_out, ns_signal, 1,
                                NVSHMEM_SIGNAL_ADD, peer_world_rank);
    }
    // Wait for gathered data to be ready
    if (tid == 0) {
      nvshmem_signal_wait_until(ns_signal, NVSHMEM_CMP_EQ, num_nodes - 1);
    }
    __syncthreads();

    for (size_t i = tid * elems_per_thd; i < num_elems_out; i += elems_per_blk) {
      // Reduce data across different nodes
      reduce_data<elems_per_thd>(reg_data, reg_acc, ns_data_out + i, num_elems_out, num_nodes);
      // Write outputs
      *reinterpret_cast<int4*>(x_out + i) = *reinterpret_cast<int4 const*>(reg_data[0]);
    }
    // Reset signals
    if (tid == 0) {
      *mem_signal = 0;
      *ns_signal = 0;
    }
  } else if constexpr (mode == MixedCommMode::OPT_BYTES) {
    // Max virtual memory buffer size:
    //   max_local_bytes * min(inter_dp_size * local_size, dp_size + 1)
    // Max nvshmem buffer size:
    //   max_local_bytes / local_tp_size * min(num_nodes * 2, inter_dp_size * 2 + 1)

    // Divide virtual memory and nvshmem buffers into blocks
    size_t num_elems_out_chunk = floor_div<local_tp_size>(max_num_elems_out);
    size_t ofst_elems_out_chunk = local_tp_rank * num_elems_out_chunk;
    size_t num_elems_mid_chunk = num_elems_out_chunk / inter_tp_size;
    size_t ofst_elems_mid_chunk = inter_tp_rank * num_elems_mid_chunk;
    size_t num_elems_mid = inter_dp_size * num_elems_out_chunk;
    T *mem_data_in, *mc_data_in, *mem_data_out, *mc_data_out;
    size_t ofst_data_local_in = dp_size * ofst_elems_out;
    if constexpr (local_tp_size > 1) {
      size_t ofst_data_local_out = dp_size * max_num_elems_out * gridDim.x + ofst_elems_out;
      mem_data_in = mem_data_all + ofst_data_local_in;
      mc_data_in = mc_data_full_all +
                   (ofst_data_local_in + local_dp_rank * max_num_elems_out + ofst_elems_out_chunk);
      mem_data_out = mem_data_all + ofst_data_local_out;
      mc_data_out = mc_data_tp_all + (ofst_data_local_out + ofst_elems_out_chunk);
    } else {
      mem_data_in = mem_data_all + ofst_data_local_in;
      mc_data_in = mc_data_full_all + (ofst_data_local_in + local_dp_rank * max_num_elems_out);
    }
    int bid_2 = bid * 2;
    uint32_t *mem_signal_in, *mc_signal_in, *mem_signal_out, *mc_signal_out;
    if constexpr (local_tp_size > 1) {
      mem_signal_in = mem_signal_all + bid_2;
      mc_signal_in = mc_signal_full_all + bid_2;
      mem_signal_out = mem_signal_in + 1;
      mc_signal_out = mc_signal_in + 1;
    } else {
      mem_signal_in = mem_signal_all + bid;
      mc_signal_in = mc_signal_full_all + bid;
    }
    T *ns_data_in, *ns_data_mid, *ns_data_mid_self, *ns_data_out, *ns_data_out_self;
    uint64_t *ns_signal_in, *ns_signal_out;
    if (inter_tp_size > 1) {
      ns_data_in = ns_data_all + bid * (inter_dp_size * 2 + 1) * num_elems_out_chunk;
      ns_data_mid = ns_data_in + num_elems_mid;
      ns_data_mid_self = ns_data_mid + node_id * num_elems_mid_chunk;
      ns_data_out = ns_data_mid + num_elems_mid;
      ns_data_out_self = ns_data_out + inter_tp_rank * num_elems_mid_chunk;
      ns_signal_in = ns_signal_all + bid_2;
      ns_signal_out = ns_signal_in + 1;
    } else {
      ns_data_in = ns_data_all + bid_2 * num_elems_mid;
      ns_data_mid = ns_data_in + num_elems_mid;
      ns_data_mid_self = ns_data_mid + node_id * num_elems_mid_chunk;
      ns_signal_in = ns_signal_all + bid;
    }

    // Read inputs
    for (size_t i = tid * elems_per_thd; i < num_elems_in; i += elems_per_blk) {
      int idx_outer = i / num_elems_out;
      size_t idx_inner = i - idx_outer * num_elems_out;
      *reinterpret_cast<int4*>(mem_data_in + (idx_outer * max_num_elems_out + idx_inner)) =
          *reinterpret_cast<int4 const*>(x_in + (idx_outer * num_elems_out_all + idx_inner));
    }
    __syncthreads();
    // Wait for inputs to be ready
    multimem_sync(mc_signal_in, mem_signal_in, local_size, tid);
    // Reduce data in the same node
    size_t local_step = local_dp_size * max_num_elems_out;
    if (inter_tp_size > 1) {
      for (size_t i = tid * elems_per_thd; i < num_elems_mid; i += elems_per_blk) {
        int idx_outer_dp = i / num_elems_out_chunk;
        size_t idx_inner_dp = i - idx_outer_dp * num_elems_out_chunk;
        int idx_outer_tp = idx_inner_dp / num_elems_mid_chunk;
        size_t idx_inner_tp = idx_inner_dp - idx_outer_tp * num_elems_mid_chunk;
        T* ns_data_val = (idx_outer_dp == inter_dp_rank && idx_outer_tp == inter_tp_rank)
                             ? ns_data_mid_self + idx_inner_tp
                             : ns_data_in + i;
        multimem_ld_reduce_vec(reg_data[0],
                               mc_data_in + (idx_outer_dp * local_step + idx_inner_dp));
        *reinterpret_cast<int4*>(ns_data_val) = *reinterpret_cast<int4 const*>(reg_data[0]);
      }
    } else {
      for (size_t i = tid * elems_per_thd; i < num_elems_mid; i += elems_per_blk) {
        int idx_outer = i / num_elems_out_chunk;
        size_t idx_inner = i - idx_outer * num_elems_out_chunk;
        T* ns_data_val =
            (idx_outer == inter_dp_rank) ? ns_data_mid_self + idx_inner : ns_data_in + i;
        multimem_ld_reduce_vec(reg_data[0], mc_data_in + (idx_outer * local_step + idx_inner));
        *reinterpret_cast<int4*>(ns_data_val) = *reinterpret_cast<int4 const*>(reg_data[0]);
      }
    }
    __syncthreads();
    // Gather data across different nodes
    size_t ns_msg_elems_mid_chunk = floor_div<elems_per_thd>(num_elems_mid_chunk);
    if (tid < num_nodes && tid != node_id) {
      int peer_world_rank = local_rank + tid * local_size;
      T const* src = ns_data_in + tid * num_elems_mid_chunk;
      nvshmem_put128_signal_nbi(ns_data_mid_self, src, ns_msg_elems_mid_chunk, ns_signal_in, 1,
                                NVSHMEM_SIGNAL_ADD, peer_world_rank);
    }
    // Wait for gathered data to be ready
    if (tid == 0) {
      nvshmem_signal_wait_until(ns_signal_in, NVSHMEM_CMP_EQ, num_nodes - 1);
    }
    __syncthreads();

    if (inter_tp_size > 1) {
      // Reduce data across different nodes
      for (size_t i = tid * elems_per_thd; i < num_elems_mid_chunk; i += elems_per_blk) {
        reduce_data<elems_per_thd>(reg_data, reg_acc, ns_data_mid + i, num_elems_mid_chunk,
                                   num_nodes);
        *reinterpret_cast<int4*>(ns_data_out_self + i) =
            *reinterpret_cast<int4 const*>(reg_data[0]);
      }
      __syncthreads();
      // Gather data across different nodes
      if (tid < inter_tp_size && tid != inter_tp_rank) {
        int peer_world_rank = local_rank + (tid + inter_dp_rank * inter_tp_size) * local_size;
        nvshmem_put128_signal_nbi(ns_data_out_self, ns_data_out_self, ns_msg_elems_mid_chunk,
                                  ns_signal_out, 1, NVSHMEM_SIGNAL_ADD, peer_world_rank);
      }
      // Wait for gathered data to be ready
      if (tid == 0) {
        nvshmem_signal_wait_until(ns_signal_out, NVSHMEM_CMP_EQ, inter_tp_size - 1);
      }
      __syncthreads();

      if constexpr (local_tp_size > 1) {
        // Gather data in the same node
        for (size_t i = tid * elems_per_thd; i < num_elems_out_chunk; i += elems_per_blk) {
          *reinterpret_cast<int4*>(reg_data[0]) = *reinterpret_cast<int4 const*>(ns_data_out + i);
          multimem_st_vec(mc_data_out + i, reg_data[0]);
        }
        __syncthreads();
      } else {
        // Write outputs
        for (size_t i = tid * elems_per_thd; i < num_elems_out; i += elems_per_blk) {
          *reinterpret_cast<int4*>(x_out + i) = *reinterpret_cast<int4 const*>(ns_data_out + i);
        }
      }
    } else {
      if constexpr (local_tp_size > 1) {
        for (size_t i = tid * elems_per_thd; i < num_elems_out_chunk; i += elems_per_blk) {
          // Reduce data across different nodes
          reduce_data<elems_per_thd>(reg_data, reg_acc, ns_data_mid + i, num_elems_out_chunk,
                                     num_nodes);
          // Gather data in the same node
          multimem_st_vec(mc_data_out + i, reg_data[0]);
        }
      } else {
        for (size_t i = tid * elems_per_thd; i < num_elems_out; i += elems_per_blk) {
          // Reduce data across different nodes
          reduce_data<elems_per_thd>(reg_data, reg_acc, ns_data_mid + i, num_elems_out_chunk,
                                     num_nodes);
          // Write outputs
          *reinterpret_cast<int4*>(x_out + i) = *reinterpret_cast<int4 const*>(reg_data[0]);
        }
      }
    }

    if constexpr (local_tp_size > 1) {
      // Wait for gathered data to be ready
      multimem_sync(mc_signal_out, mem_signal_out, local_size, tid);
      // Write outputs
      for (size_t i = tid * elems_per_thd; i < num_elems_out; i += elems_per_blk) {
        *reinterpret_cast<int4*>(x_out + i) = *reinterpret_cast<int4 const*>(mem_data_out + i);
      }
    }
    // Reset signals
    if (tid == 0) {
      if constexpr (local_tp_size > 1) {
        *reinterpret_cast<uint64_t*>(mem_signal_in) = 0;
      } else {
        *mem_signal_in = 0;
      }
      if (inter_tp_size > 1) {
        *reinterpret_cast<__uint128_t*>(ns_signal_in) = 0;
      } else {
        *ns_signal_in = 0;
      }
    }
  } else {
    static_assert(mode == MixedCommMode::MIXED, "Unsupported mode");
    // Max virtual memory buffer size:
    //   max_local_bytes * min(inter_dp_size * local_size, dp_size + 1)
    // Max nvshmem buffer size:
    //   max_local_bytes / local_tp_size * (inter_dp_size + num_nodes)

    // Divide virtual memory and nvshmem buffers into blocks
    size_t num_elems_out_chunk = floor_div<local_tp_size>(max_num_elems_out);
    size_t ofst_elems_out_chunk = local_tp_rank * num_elems_out_chunk;
    size_t num_elems_mid = inter_dp_size * num_elems_out_chunk;
    T *mem_data_in, *mc_data_in, *mem_data_out, *mc_data_out;
    size_t ofst_data_local_in = dp_size * ofst_elems_out;
    if constexpr (local_tp_size > 1) {
      size_t ofst_data_local_out = dp_size * max_num_elems_out * gridDim.x + ofst_elems_out;
      mem_data_in = mem_data_all + ofst_data_local_in;
      mc_data_in = mc_data_full_all +
                   (ofst_data_local_in + local_dp_rank * max_num_elems_out + ofst_elems_out_chunk);
      mem_data_out = mem_data_all + ofst_data_local_out;
      mc_data_out = mc_data_tp_all + (ofst_data_local_out + ofst_elems_out_chunk);
    } else {
      mem_data_in = mem_data_all + ofst_data_local_in;
      mc_data_in = mc_data_full_all + (ofst_data_local_in + local_dp_rank * max_num_elems_out);
    }
    T* ns_data_in = ns_data_all + bid * (inter_dp_size + num_nodes) * num_elems_out_chunk;
    T* ns_data_out = ns_data_in + num_elems_mid;
    T* ns_data_out_self = ns_data_out + node_id * num_elems_out_chunk;
    uint32_t *mem_signal_in, *mc_signal_in, *mem_signal_out, *mc_signal_out;
    if constexpr (local_tp_size > 1) {
      int bid_2 = bid * 2;
      mem_signal_in = mem_signal_all + bid_2;
      mc_signal_in = mc_signal_full_all + bid_2;
      mem_signal_out = mem_signal_in + 1;
      mc_signal_out = mc_signal_in + 1;
    } else {
      mem_signal_in = mem_signal_all + bid;
      mc_signal_in = mc_signal_full_all + bid;
    }
    uint64_t* ns_signal = ns_signal_all + bid;

    // Read inputs
    for (size_t i = tid * elems_per_thd; i < num_elems_in; i += elems_per_blk) {
      int idx_outer = i / num_elems_out;
      size_t idx_inner = i - idx_outer * num_elems_out;
      *reinterpret_cast<int4*>(mem_data_in + (idx_outer * max_num_elems_out + idx_inner)) =
          *reinterpret_cast<int4 const*>(x_in + (idx_outer * num_elems_out_all + idx_inner));
    }
    __syncthreads();
    // Wait for inputs to be ready
    multimem_sync(mc_signal_in, mem_signal_in, local_size, tid);
    // Reduce data in the same node
    size_t local_step = local_dp_size * max_num_elems_out;
    for (size_t i = tid * elems_per_thd; i < num_elems_mid; i += elems_per_blk) {
      int idx_outer = i / num_elems_out_chunk;
      size_t idx_inner = i - idx_outer * num_elems_out_chunk;
      T* ns_data_val = (idx_outer == inter_dp_rank) ? ns_data_out_self + idx_inner : ns_data_in + i;
      multimem_ld_reduce_vec(reg_data[0], mc_data_in + idx_outer * local_step + idx_inner);
      *reinterpret_cast<int4*>(ns_data_val) = *reinterpret_cast<int4 const*>(reg_data[0]);
    }
    __syncthreads();

    // Gather data across different nodes
    size_t ns_msg_elems_out_chunk = floor_div<elems_per_thd>(num_elems_out_chunk);
    if (tid < num_nodes && tid != node_id) {
      int peer_world_rank = local_rank + tid * local_size;
      int peer_inter_dp_rank = tid / inter_tp_size;
      T const* src = (peer_inter_dp_rank == inter_dp_rank)
                         ? ns_data_out_self
                         : ns_data_in + peer_inter_dp_rank * num_elems_out_chunk;
      nvshmem_put128_signal_nbi(ns_data_out_self, src, ns_msg_elems_out_chunk, ns_signal, 1,
                                NVSHMEM_SIGNAL_ADD, peer_world_rank);
    }
    // Wait for gathered data to be ready
    if (tid == 0) {
      nvshmem_signal_wait_until(ns_signal, NVSHMEM_CMP_EQ, num_nodes - 1);
    }
    __syncthreads();

    if constexpr (local_tp_size > 1) {
      for (size_t i = tid * elems_per_thd; i < num_elems_out_chunk; i += elems_per_blk) {
        // Reduce data across different nodes
        reduce_data<elems_per_thd>(reg_data, reg_acc, ns_data_out + i, num_elems_out_chunk,
                                   num_nodes);
        // Gather data in the same node
        multimem_st_vec(mc_data_out + i, reg_data[0]);
      }
      __syncthreads();
      // Wait for gathered data to be ready
      multimem_sync(mc_signal_out, mem_signal_out, local_size, tid);
      // Write outputs
      for (size_t i = tid * elems_per_thd; i < num_elems_out; i += elems_per_blk) {
        *reinterpret_cast<int4*>(x_out + i) = *reinterpret_cast<int4 const*>(mem_data_out + i);
      }
    } else {
      for (size_t i = tid * elems_per_thd; i < num_elems_out; i += elems_per_blk) {
        // Reduce data across different nodes
        reduce_data<elems_per_thd>(reg_data, reg_acc, ns_data_out + i, num_elems_out_chunk,
                                   num_nodes);
        // Write outputs
        *reinterpret_cast<int4*>(x_out + i) = *reinterpret_cast<int4 const*>(reg_data[0]);
      }
    }
    // Reset signals
    if (tid == 0) {
      if constexpr (local_tp_size > 1) {
        *reinterpret_cast<uint64_t*>(mem_signal_in) = 0;
      } else {
        *mem_signal_in = 0;
      }
      *ns_signal = 0;
    }
  }
}

}  // namespace mixed_comm
}  // namespace flashinfer
